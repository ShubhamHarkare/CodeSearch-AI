"""
CodeSearch AI - Streamlit Frontend
A professional chat interface for React documentation search
"""

import streamlit as st
from chatbot import ReactDotChatbot
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CodeSearch AI",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main chat container */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Code blocks */
    .stCodeBlock {
    background-color: #0d1117;
    color: #c9d1d9;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    border: 1px solid #30363d;
    }   
    
    /* Metrics styling */
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Sidebar */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Cache badge */
    .cache-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    .cache-hit {
        background-color: #d4edda;
        color: #155724;
    }
    
    .cache-miss {
        background-color: #fff3cd;
        color: #856404;
    }
    
    /* Example buttons */
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Initialize chatbot (expensive operation, do once)
if 'chatbot' not in st.session_state:
    with st.spinner("🚀 Initializing CodeSearch AI..."):
        st.session_state.chatbot = ReactDotChatbot(use_cache=True)

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize query counter
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_response_time(seconds):
    """Format response time with color coding"""
    if seconds < 0.1:
        color = "#28a745"  # Green
        label = "⚡ Instant"
    elif seconds < 1.0:
        color = "#28a745"  # Green
        label = "🟢 Fast"
    elif seconds < 3.0:
        color = "#ffc107"  # Yellow
        label = "🟡 Good"
    else:
        color = "#dc3545"  # Red
        label = "🔴 Slow"
    
    return f'<span style="color: {color};">{label}: {seconds:.3f}s</span>'

def display_sources(sources):
    """Display sources in a clean format"""
    if not sources:
        st.caption("ℹ️ No sources available")
        return
    
    with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
        for i, source in enumerate(sources, 1):
            # Extract filename from path
            filename = source.split('/')[-1] if '/' in source else source
            st.markdown(f"**{i}.** `{filename}`")
            st.caption(f"📄 {source}")

def process_query(prompt):
    """Process user query and return formatted response"""
    # Get response from chatbot
    with st.spinner("🔍 Searching React documentation..."):
        response = st.session_state.chatbot.ask(prompt)
    
    return response

# ============================================================================
# SIDEBAR - METRICS & SETTINGS
# ============================================================================

with st.sidebar:
    st.title("📊 Dashboard")
    
    # System health check
    with st.expander("🏥 System Health", expanded=False):
        health = st.session_state.chatbot.health_check()
        if health['status'] == 'healthy':
            st.success("✅ All systems operational")
        else:
            st.warning(f"⚠️ Status: {health['status']}")
        
        for component, status in health['components'].items():
            status_icon = "🟢" if status.get('status') == 'healthy' else "🔴"
            st.caption(f"{status_icon} {component.upper()}: {status.get('status', 'unknown')}")
    
    st.divider()
    
    # Performance Metrics
    st.subheader("⚡ Performance")
    
    # Get stats
    cache_stats = st.session_state.chatbot.get_cache_stats()
    log_stats = st.session_state.chatbot.get_logs_summary()
    
    # Display metrics in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Total Queries",
            st.session_state.total_queries,
            help="Total questions asked this session"
        )
        
        st.metric(
            "Cache Hits",
            cache_stats.get('hits', 0),
            help="Queries answered from cache"
        )
    
    with col2:
        hit_rate = cache_stats.get('hit_rate_percent', 0)
        st.metric(
            "Hit Rate",
            f"{hit_rate:.1f}%",
            help="Percentage of cached responses"
        )
        
        avg_time = log_stats.get('avg_response_time', 0)
        st.metric(
            "Avg Time",
            f"{avg_time:.2f}s",
            help="Average response time"
        )
    
    # Cache status indicator
    st.divider()
    if hit_rate > 40:
        st.success(f"🔥 Cache is hot! {hit_rate:.1f}% hit rate")
    elif hit_rate > 20:
        st.info(f"⚡ Cache warming up: {hit_rate:.1f}%")
    elif st.session_state.total_queries > 0:
        st.warning(f"❄️ Cache is cold: {hit_rate:.1f}%")
    else:
        st.info("💤 No queries yet - cache is idle")
    
    # Additional metrics
    if st.session_state.total_queries > 0:
        st.divider()
        st.caption("📈 Detailed Stats")
        
        fastest = log_stats.get('fastest_response', 0)
        slowest = log_stats.get('slowest_response', 0)
        
        st.caption(f"⚡ Fastest: {fastest:.3f}s")
        st.caption(f"🐌 Slowest: {slowest:.2f}s")
        st.caption(f"📦 Cached Queries: {cache_stats.get('cached_queries', 0)}")
    
    # Popular queries
    if st.session_state.total_queries > 2:
        st.divider()
        st.caption("🔥 Popular Queries")
        popular = st.session_state.chatbot.get_popular_queries(top_n=3)
        for i, item in enumerate(popular, 1):
            query_preview = item['query'][:30] + "..." if len(item['query']) > 30 else item['query']
            st.caption(f"{i}. {query_preview} ({item['count']}x)")
    
    # Settings
    st.divider()
    st.subheader("⚙️ Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("💾 Clear Cache", use_container_width=True):
            result = st.session_state.chatbot.clear_cache()
            st.success(f"Cleared {result.get('deleted', 0)} queries!")
            time.sleep(1)
            st.rerun()
    
    # Footer
    st.divider()
    st.caption("Built with ❤️ using Streamlit")
    st.caption("Powered by Groq + LangChain + Redis")

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Header
st.title("⚛️ CodeSearch AI")
st.caption("🔍 Semantic search for React documentation powered by AI")

# Welcome message (only show if no messages)
if len(st.session_state.messages) == 0:
    st.info("""
    👋 **Welcome to CodeSearch AI!**
    
    Ask me anything about React - I'll search through 500+ pages of official documentation 
    and provide accurate answers with working code examples.
    
    Try the example questions below or type your own question!
    """)

# Example questions
st.markdown("💡 **Quick Start - Try these:**")

col1, col2, col3, col4 = st.columns(4)

example_questions = [
    ("🎯 useState hook?", "How do I use the useState hook?"),
    ("📝 What is JSX?", "What is JSX and why do we use it?"),
    ("✅ Form validation?", "How do I handle form validation in React?"),
    ("⚡ React.memo?", "Show me how to optimize performance with React.memo")
]

for col, (btn_text, question) in zip([col1, col2, col3, col4], example_questions):
    with col:
        if st.button(btn_text, use_container_width=True):
            # Add question to chat
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            
            # Get response
            response = process_query(question)
            
            # Add response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response['answer'],
                "sources": response['sources'],
                "cached": response['cached'],
                "response_time": response['response_time']
            })
            
            # Update counter
            st.session_state.total_queries += 1
            
            # Rerun to show message
            st.rerun()

st.divider()

# Display chat history
for message in st.session_state.messages:
    if message['role'] == 'user':
        with st.chat_message("user", avatar="👤"):
            st.markdown(message['content'])
    
    else:
        with st.chat_message("assistant", avatar="🤖"):
            # Display answer
            st.markdown(message['content'])
            
            # Display sources
            if 'sources' in message:
                display_sources(message['sources'])
            
            # Display metadata (cache status + response time)
            if 'cached' in message and 'response_time' in message:
                cached = message['cached']
                response_time = message['response_time']
                
                # Create badge
                badge_class = "cache-hit" if cached else "cache-miss"
                cache_icon = "💾" if cached else "🔍"
                cache_text = "Cached" if cached else "Fresh"
                
                time_html = format_response_time(response_time)
                
                st.markdown(
                    f'<div class="cache-badge {badge_class}">{cache_icon} {cache_text}</div> '
                    f'{time_html}',
                    unsafe_allow_html=True
                )

# Chat input
if prompt := st.chat_input("Ask me anything about React...", key="chat_input"):
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Get bot response
    response = process_query(prompt)
    
    # Add assistant message to chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": response['answer'],
        "sources": response['sources'],
        "cached": response['cached'],
        "response_time": response['response_time']
    })
    
    # Display assistant message
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(response['answer'])
        display_sources(response['sources'])
        
        # Metadata badge
        cached = response['cached']
        response_time = response['response_time']
        
        badge_class = "cache-hit" if cached else "cache-miss"
        cache_icon = "💾" if cached else "🔍"
        cache_text = "Cached" if cached else "Fresh"
        
        time_html = format_response_time(response_time)
        
        st.markdown(
            f'<div class="cache-badge {badge_class}">{cache_icon} {cache_text}</div> '
            f'{time_html}',
            unsafe_allow_html=True
        )
    
    # Update query counter
    st.session_state.total_queries += 1
    
    # Rerun to update sidebar metrics
    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.caption("🎯 **CodeSearch AI** - Making React docs searchable with AI")

with footer_col2:
    if st.session_state.total_queries > 0:
        st.caption(f"💬 {st.session_state.total_queries} queries answered")

with footer_col3:
    cache_stats = st.session_state.chatbot.get_cache_stats()
    hit_rate = cache_stats.get('hit_rate_percent', 0)
    if st.session_state.total_queries > 0:
        st.caption(f"💾 {hit_rate:.0f}% cache hit rate")