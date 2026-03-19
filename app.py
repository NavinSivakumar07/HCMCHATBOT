import streamlit as st
import time
import asyncio
from Chatbot import query_schema, get_global_hr_table_urls, populate_collection

# --- Page Configuration ---
st.set_page_config(
    page_title="HCM Data Explorer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling (Matching New Screenshot) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #fcfcfd;
    }
    
    /* Center Layout */
    .chat-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding-top: 30px;
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #004b7c; /* Match Company Logo Blue */
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 2.5rem;
    }
    
    /* Top Main Action Buttons */
    .main-grid {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin-bottom: 2rem;
    }
    
    /* Quick Help Section */
    .section-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Card Styles for Grid */
    .stButton > button {
        background-color: white !important;
        border: 1px solid #e2e8f0 !important;
        color: #334155 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
        height: auto !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
    }
    
    .stButton > button:hover {
        border-color: #004b7c !important; /* Blue border on hover */
        background-color: #f8fafc !important;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05) !important;
    }

    /* Target the text input area */
    div[data-testid="stChatInput"] {
        border-radius: 20px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recent_searches" not in st.session_state:
    st.session_state.recent_searches = ["PER_ALL_PEOPLE_F", "PAY_PAYROLL_ACTIONS", "PER_ALL_PEOPLE_F"]

# Helper to run async tasks
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

async def init_data_task():
    if "initialized" not in st.session_state:
        st.write("Initializing HCM Knowledge Base...")
        toc_url = 'https://docs.oracle.com/en/cloud/saas/human-resources/oedmh/toc.htm'
        urls = get_global_hr_table_urls(toc_url)
        if urls:
            await populate_collection(urls[:10])
        st.session_state.initialized = True
        st.rerun()

# --- Page Content ---
if not st.session_state.messages:
    # --- Landing Page Layout ---
    cols_center = st.columns([1, 4, 1])
    with cols_center[1]:
        # Center the logo using st.image
        logo_cols = st.columns([1, 1, 1])
        with logo_cols[1]:
            st.image("image/logo.jpg", use_container_width=True)
            
        st.markdown("""
        <div class="chat-container">
            <div class="hero-title">HCM Data Explorer & Query Assistant</div>
            <div class="hero-subtitle">Locate HCM Tables, Columns, and Generate Complex Queries. Your data, accessible.</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. Main Action Buttons
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            if st.button(" Find HCM Table Names", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "List some important HCM table names for HR."})
                st.rerun()
        with m_col2:
            if st.button(" Build HR Reports Query", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "How do I build a query for an HR report using HCM tables?"})
                st.rerun()

        st.markdown("<div style='text-align: center;' class='section-title'>Quick Help Suggestions</div>", unsafe_allow_html=True)
        
        # 2. Suggestions Grid
        g1, g2 = st.columns(2)
        with g1:
            if st.button(" Find Payroll Table", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Find payroll related tables."})
                st.rerun()
            if st.button("List Employees by Dept Query", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Query to list all employees by department."})
                st.rerun()
            if st.button("Employee Performance Data", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Where is employee performance history data found?"})
                st.rerun()
        with g2:
            if st.button(" Schema for Benefits", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Show me the schema for benefits tables."})
                st.rerun()
            if st.button(" View Timekeeping Columns", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "What are the common columns in timekeeping tables?"})
                st.rerun()
            if st.button(" Check Leave Accruals Table", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": "Which table stores leave accruals?"})
                st.rerun()

else:
    # --- Chat History Mode ---
    st.markdown("<h3 style='text-align: center; color: #004b7c;'>HCM Data Assistant</h3>", unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Generate Assistant Response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Searching HCM knowledge base... ")
            
            try:
                prompt = st.session_state.messages[-1]["content"]
                answer = query_schema(prompt)
                message_placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                # Add to recent searches (simplified for demo)
                if len(prompt) < 30: st.session_state.recent_searches.insert(0, prompt)
            except Exception as e:
                message_placeholder.error(f"Error: {e}")

# --- Floating Sidebar for "Recent Searches" ---
with st.sidebar:
    st.markdown("### Recent Table Searches")
    for search in st.session_state.recent_searches[:10]:
        st.markdown(f" `{search}`")

# --- Chat Input ---
if prompt := st.chat_input("Type message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# --- Database Initialization ---
if "initialized" not in st.session_state:
    run_async(init_data_task())
