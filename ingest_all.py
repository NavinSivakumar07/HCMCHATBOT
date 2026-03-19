import asyncio
from Chatbot import get_global_hr_table_urls, populate_collection, index

async def main():
    # Step 1: Clear existing (mixed) data from Pinecone 
    stats = index.describe_index_stats()
    print(f"Current vectors in Pinecone: {stats['total_vector_count']}")
    
    if stats['total_vector_count'] > 0:
        print("Clearing old data from index...")
        index.delete(delete_all=True)
        print("Index cleared.")
    
    # Step 2: Get ONLY Global Human Resources table URLs
    print("\nFetching Global Human Resources table URLs...")
    toc_url = 'https://docs.oracle.com/en/cloud/saas/human-resources/oedmh/toc.htm'
    all_urls = get_global_hr_table_urls(toc_url)
    
    print(f"Discovered {len(all_urls)} Global Human Resources tables.")
    
    if not all_urls:
        print("No URLs found. Check your internet connection or the source URL.")
        return

    # Step 3: Start Batched Ingestion (ALL Global HR tables)
    await populate_collection(all_urls, batch_size=20)
    
    print(f"\n=== DONE ===")
    print(f"All {len(all_urls)} Global HR tables have been processed and stored in Pinecone.")

if __name__ == "__main__":
    asyncio.run(main())
