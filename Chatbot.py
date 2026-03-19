import os
import asyncio
import numpy as np 
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone
from groq import Groq
from google import genai
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler, CacheMode

# Load environment variables
load_dotenv()

# Initialize clients using the loaded .env values
groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_host = os.getenv("PINECONE_HOST")

# Groq and Gemini clients
groq_client = Groq(api_key=groq_api_key)
client_gemini = genai.Client(api_key=gemini_api_key)

# Pinecone client initialization
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(host=pinecone_host)

# Max characters to send to Gemini embedding (avoids 500 errors on huge pages)
MAX_EMBED_CHARS = 8000

def get_global_hr_table_urls(toc_url):
    """Scrapes only Global Human Resources table URLs."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(toc_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    urls = []
    base_url = 'https://docs.oracle.com/en/cloud/saas/human-resources/oedmh/'
    global_hr = None
    for span in soup.find_all('span'):
        if "Global Human Resources" in span.get_text():
            global_hr = span
            break
    if not global_hr:
        return []
    
    x = global_hr.find_next_sibling('ul')
    for i in x.find_all('a', href=True):
        href = i['href'].split('#')[0]
        if href.endswith('vl') or href.endswith('v') or not href.startswith('per'):
            continue
        link = base_url + href
        if link not in urls:
            urls.append(link)
    return urls

def get_all_hcm_table_urls(toc_url):
    """Scrapes ALL table URLs in the documentation (all categories)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(toc_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    
    urls = []
    base_url = 'https://docs.oracle.com/en/cloud/saas/human-resources/oedmh/'
    
    for ul in soup.find_all('ul'):
        for i in ul.find_all('a', href=True):
            href = i['href'].split('#')[0]
            if href.endswith('.htm') or href.endswith('.html'):
                if href.startswith('per') or href.startswith('pay') or href.startswith('hcm') or \
                   href.startswith('cmp') or href.startswith('ben') or href.startswith('ghr'):
                    link = base_url + href
                    if link not in urls:
                        urls.append(link)
    return urls

async def scrape_tables_with_crawl4ai(urls):
    """Leverages Crawl4AI to scrape multiple URLs asynchronously."""
    if not urls:
        return []
    
    print(f"Crawl4AI: Scraping {len(urls)} tables...")
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun_many(
            urls=urls,
            cache_mode=CacheMode.ENABLED
        )
        
        extracted_data = []
        for result in results:
            if result.success:
                table_content = result.markdown
                table_name = result.metadata.get('title', urls[results.index(result)].split('/')[-1])
                extracted_data.append({'Table_Name': table_name, 'Text': table_content})
        return extracted_data

async def populate_collection(urls_to_process, batch_size=20):
    """Populates Pinecone with batching, rate limiting, and content truncation."""
    stats = index.describe_index_stats()
    existing_count = stats['total_vector_count']
    total_urls = len(urls_to_process)
    
    print(f"Pinecone has {existing_count} vectors. Tables to process: {total_urls}.")
    
    if existing_count >= total_urls:
        print("Index appears to be fully populated. Skipping.")
        return

    for start_idx in range(existing_count, total_urls, batch_size):
        end_idx = min(start_idx + batch_size, total_urls)
        batch_urls = urls_to_process[start_idx:end_idx]
        
        print(f"\n--- Batch {start_idx}-{end_idx} ---")
        batch_data = await scrape_tables_with_crawl4ai(batch_urls)
        
        vectors_to_upsert = []
        for i, chunk in enumerate(batch_data):
            for attempt in range(3):
                try:
                    await asyncio.sleep(1.5 + (attempt * 3))
                    
                    # TRUNCATE content to avoid Gemini 500 errors on huge pages
                    text_to_embed = chunk['Text'][:MAX_EMBED_CHARS]
                    text_for_metadata = chunk['Text'][:MAX_EMBED_CHARS]
                    
                    print(f"  [{start_idx + i}] {chunk['Table_Name']} ({len(chunk['Text'])} chars, truncated to {len(text_to_embed)}) Attempt {attempt+1}")
                    
                    result = client_gemini.models.embed_content(
                        model="gemini-embedding-001", 
                        contents=text_to_embed
                    )
                    vectors_to_upsert.append({
                        "id": f"table_{start_idx + i}",
                        "values": result.embeddings[0].values,
                        "metadata": {"Table_Name": chunk["Table_Name"], "text": text_for_metadata}
                    })
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  FAILED {chunk['Table_Name']} after 3 attempts: {e}")
                    else:
                        print(f"  Retry {chunk['Table_Name']}: {e}")
        
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            print(f"  ✓ Batch {start_idx}-{end_idx}: {len(vectors_to_upsert)} vectors upserted.")
        await asyncio.sleep(3)
    
    print("\n=== Ingestion Complete! ===")

def query_schema(ip):
    """Queries the index and triggers real-time backup if match is weak."""
    query = client_gemini.models.embed_content(model="gemini-embedding-001", contents=ip)
    query_vector = query.embeddings[0].values
    
    response = index.query(vector=query_vector, top_k=2, include_metadata=True)
    
    # SCORE THRESHOLD: If match score is too low, trigger real-time crawl
    THRESHOLD = 0.5
    if not response['matches'] or response['matches'][0]['score'] < THRESHOLD:
        print(f"Low score ({response['matches'][0]['score'] if response['matches'] else 'None'}). Triggering search backup...")
        return dynamic_search_and_answer(ip)

    retrieved_txt = ""
    retrieved_table = []
    for match in response['matches']:
        retrieved_txt += match['metadata']['text'] + "\n"
        retrieved_table.append(match['metadata']['Table_Name'])
    
    response_completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"You are an Oracle HCM expert. Use the following: {retrieved_table}\nDocumentation:\n{retrieved_txt}\nQuestion: {ip}"}]
    )
    return response_completion.choices[0].message.content

def dynamic_search_and_answer(query):
    """SEARCH BACKUP: Searches TOC for potential table names, crawls on-the-fly."""
    toc_url = 'https://docs.oracle.com/en/cloud/saas/human-resources/oedmh/toc.htm'
    all_urls = get_all_hcm_table_urls(toc_url)
    
    query_upper = query.upper()
    potential_urls = []
    for url in all_urls:
        filename = url.split('/')[-1].replace('.html', '').replace('.htm', '').upper()
        if any(word in filename for word in query_upper.split()):
            potential_urls.append(url)
    
    if not potential_urls:
        return "I couldn't find a strong match in the database or identify any relevant table names from the documentation to search on-the-fly."

    print(f"Searching {len(potential_urls[:3])} table pages on the fly...")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(scrape_tables_with_crawl4ai(potential_urls[:3]))
    except:
        return "I tried to search the documentation on-the-fly but encountered a technical error."

    text_to_use = "\n".join([d['Text'][:MAX_EMBED_CHARS] for d in data])
    
    response_completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": f"I searched the Oracle HCM documentation on-the-fly. Using this information:\n{text_to_use}\nAnswer the question: {query}"}]
    )
    return response_completion.choices[0].message.content

async def main():
    toc_url = 'https://docs.oracle.com/en/cloud/saas/human-resources/oedmh/toc.htm'
    res = get_global_hr_table_urls(toc_url)
    print(f"Found {len(res)} table URLs.")
    
    if res:
        await populate_collection(res[:10])
    
    print("\n--- Chatbot Query Result ---")
    question = "Where is SCH_BASED_DUR found?"
    print(f"Question: {question}")
    answer = query_schema(question)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    asyncio.run(main())
