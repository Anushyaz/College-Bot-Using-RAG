"""
Streamlit Web Application: AI-Powered College Regulations and Academic Policy Chatbot Using RAG
Featuring a navy blue / gold theme, PDF Processing, Vector Retrieval,
Conversational History, Grounded Citations, and a Live Evaluation Dashboard.
"""

import os
import io
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

# Load environment configuration
load_dotenv()

from document_processor import extract_pages_from_pdf, process_document, DocumentChunk
from rag_pipeline import CollegeRAGPipeline, POLICY_FALLBACK_RESPONSE, OFFICIAL_DISCLAIMER
from evaluation import run_evaluation_suite, BENCHMARK_DATASET

# Page configuration
st.set_page_config(
    page_title="College Regulations RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom navy blue and gold CSS styling
CUSTOM_CSS = """
<style>
    /* Navy surfaces with warm gold accents */
    :root {
        --bg-main: #081426;
        --bg-surface: #10223a;
        --bg-surface-elevated: #19314f;
        --border-color: #29415e;
        --text-primary: #edf2f8;
        --text-secondary: #a8b6c9;
        --gold-primary: #d7ad55;
        --gold-hover: #edc66f;
        --gold-light: #edc66f;
        --gold-bg: rgba(215, 173, 85, 0.14);
    }

    /* App-wide spacing, surfaces, and interactive controls */
    .stApp {
        background:
            radial-gradient(ellipse at 88% 0%, rgba(215, 173, 85, 0.08), transparent 34%),
            var(--bg-main);
        color: var(--text-primary);
    }
    [data-testid="stAppViewContainer"] {
        background: transparent;
    }
    [data-testid="stHeader"] {
        background: rgba(8, 20, 38, 0.92);
        color: var(--text-primary);
    }
    [data-testid="stToolbar"] {
        color: var(--text-primary);
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stSidebar"] {
        background-color: #0b1a2e;
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary);
    }
    [role="tablist"] {
        gap: 8px;
        border-bottom: 1px solid var(--border-color);
    }
    [data-testid="stTab"] {
        color: var(--text-secondary) !important;
        border-radius: 8px 8px 0 0;
        padding: 10px 14px;
        transition: color 0.2s ease, background-color 0.2s ease;
    }
    [data-testid="stTab"]:hover {
        color: var(--text-primary) !important;
        background-color: rgba(215, 173, 85, 0.08) !important;
    }
    [data-testid="stTab"][aria-selected="true"] {
        color: var(--gold-primary) !important;
        background-color: rgba(215, 173, 85, 0.1) !important;
        box-shadow: inset 0 -2px 0 var(--gold-primary);
    }
    .react-aria-SelectionIndicator {
        background-color: var(--gold-primary) !important;
    }
    .stButton > button {
        min-height: 2.75rem;
        border: 1px solid var(--border-color) !important;
        border-radius: 9px;
        white-space: normal;
        line-height: 1.35;
        transition: border-color 0.2s ease, background-color 0.2s ease, transform 0.2s ease;
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
    }
    .stButton > button p {
        color: inherit !important;
    }
    .stButton > button:hover {
        border-color: var(--gold-primary) !important;
        color: var(--gold-primary) !important;
        background-color: var(--bg-surface-elevated) !important;
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        border-color: var(--gold-primary) !important;
        background: linear-gradient(135deg, #edc66f, #c99b3e) !important;
        color: #10223a !important;
    }
    .stButton > button[kind="primary"] p {
        color: #10223a !important;
    }
    [data-testid="stChatInput"] {
        border: 1px solid var(--border-color);
        border-radius: 12px;
        background-color: #0d1c31;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.16);
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text-primary);
        caret-color: var(--gold-primary);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--gold-primary);
        box-shadow: 0 0 0 2px rgba(215, 173, 85, 0.16);
    }
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(215, 173, 85, 0.55);
        border-radius: 12px;
        background-color: rgba(16, 34, 58, 0.65);
        padding: 8px;
    }
    [data-testid="stExpander"] {
        border-color: var(--border-color);
        border-radius: 10px;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--border-color);
        border-radius: 14px;
        background-color: rgba(16, 34, 58, 0.78);
        padding: 1rem 1.1rem;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid var(--border-color);
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stAlert"] {
        border-radius: 10px;
    }
    a {
        color: var(--gold-primary);
    }
    
    /* Header hero */
    .hero-container {
        background:
            radial-gradient(ellipse at 88% 15%, rgba(215, 173, 85, 0.18), transparent 36%),
            linear-gradient(120deg, #0d1c31 0%, #142b49 58%, #1b3552 100%);
        border: 1px solid rgba(215, 173, 85, 0.3);
        border-radius: 16px;
        padding: 30px 34px;
        margin-bottom: 28px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.24);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #b8c5d6;
        line-height: 1.5;
        max-width: 760px;
    }
    .status-card {
        background: linear-gradient(145deg, #122743, #0f2037);
        border: 1px solid var(--border-color);
        border-radius: 11px;
        padding: 13px 14px;
        margin: 8px 0;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.12);
    }
    .status-label {
        color: var(--text-secondary);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 5px;
    }
    .status-value {
        color: var(--text-primary);
        font-size: 14px;
        font-weight: 600;
    }
    .status-note {
        color: var(--text-secondary);
        font-size: 11px;
        margin-top: 7px;
    }

    /* Stat cards */
    .metric-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: var(--gold-primary);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--gold-light);
        margin: 4px 0;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
    }

    /* Citation badge */
    .citation-badge {
        display: inline-block;
        background: var(--gold-bg);
        color: var(--gold-primary);
        border: 1px solid rgba(215, 173, 85, 0.38);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    /* Chat bubble enhancements */
    .chat-timestamp {
        font-size: 10px;
        color: var(--text-secondary);
        margin-top: 4px;
    }
    
    /* Disclaimer footer */
    .disclaimer-box {
        background-color: #10223a;
        border-left: 4px solid var(--gold-primary);
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 12px;
        color: #c3cedb;
        margin-top: 20px;
    }
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 2rem;
        }
        .hero-container {
            padding: 20px;
        }
        .hero-title {
            font-size: 21px;
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Pipeline in Streamlit Session State
@st.cache_resource
def get_rag_pipeline():
    """Caches pipeline instance across UI re-runs to avoid re-loading embedding weights."""
    return CollegeRAGPipeline()


pipeline = get_rag_pipeline()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_docs_meta" not in st.session_state:
    st.session_state.uploaded_docs_meta = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def auto_load_sample_if_empty():
    """Pre-indexes sample regulations PDF if the vector store is currently blank."""
    if len(pipeline.chunks) == 0:
        sample_pdf = os.path.join("data", "sample_college_regulations.pdf")
        if not os.path.exists(sample_pdf):
            try:
                from sample_data_generator import generate_sample_pdf
                generate_sample_pdf(sample_pdf)
            except Exception as e:
                return
        if os.path.exists(sample_pdf):
            try:
                chunks = process_document(sample_pdf, filename="sample_college_regulations.pdf")
                pipeline.build_index(chunks)
                st.session_state.uploaded_docs_meta = [{
                    "filename": "sample_college_regulations.pdf",
                    "pages": 3,
                    "chunks": len(chunks),
                    "added_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                }]
            except Exception:
                pass


auto_load_sample_if_empty()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### 🎓 Academic Policy Bot")
    st.markdown(
        "<small style='color:#a8b6c9'>Powered by Retrieval-Augmented Generation (RAG) & Sentence Transformers</small>",
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown("#### ⚡ System Status")
    chunks_count = len(pipeline.chunks)
    llm_mode = (
        "OpenAI API"
        if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_api_key_here"
        else "Local extractive engine"
    )
    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">Knowledge base</div>
            <div class="status-value">{'🟢 Ready' if chunks_count > 0 else '🟡 No documents indexed'}</div>
            <div class="status-note">{chunks_count} searchable text chunks</div>
        </div>
        <div class="status-card">
            <div class="status-label">Answer engine</div>
            <div class="status-value">{llm_mode}</div>
            <div class="status-note">Embeddings: all-MiniLM-L6-v2</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    
    st.markdown("#### 🛠️ Vector Store Controls")
    if st.button("🔄 Reset Index", use_container_width=True):
        pipeline.clear_index()
        st.session_state.uploaded_docs_meta = []
        st.session_state.messages = []
        st.success("Vector index cleared.")
        st.rerun()
    if st.button("📚 Load Sample Regulations", use_container_width=True):
        sample_pdf = os.path.join("data", "sample_college_regulations.pdf")
        if not os.path.exists(sample_pdf):
            from sample_data_generator import generate_sample_pdf
            generate_sample_pdf(sample_pdf)
        chunks = process_document(sample_pdf, filename="sample_college_regulations.pdf")
        pipeline.build_index(chunks)
        st.session_state.uploaded_docs_meta = [{
            "filename": "sample_college_regulations.pdf",
            "pages": 3,
            "chunks": len(chunks),
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }]
        st.success(f"Indexed {len(chunks)} chunks from Sample Regulations.")
        st.rerun()

    st.divider()
    
    st.markdown("#### ⚖️ Regulatory Notice")
    st.info(
        "Answers are grounded exclusively in uploaded official documents. "
        "The Office of the Academic Registrar remains the sole final authority for official interpretations."
    )


# ==================== MAIN HERO HEADER ====================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">
            <span>🏛️</span> College Policy Assistant
        </div>
        <div class="hero-subtitle">
            Instant, evidence-grounded answers to academic questions regarding attendance requirements, 
            examination guidelines, grading, fee deadlines, transcripts, and leave procedures.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Navigation Tabs
tab_chat, tab_admin, tab_eval, tab_guide = st.tabs([
    "💬 Academic Policy Chat",
    "📁 Document Library & Upload",
    "📊 RAG Evaluation Suite",
    "ℹ️ Architecture & System Guide"
])

# ==================== TAB 1: CHAT INTERFACE ====================
with tab_chat:
    # Example Questions Section
    st.markdown("##### 💡 Start with a common question")
    st.caption("Select a prompt to ask it, or write your own question in the chat box.")
    example_cols = st.columns(2)
    
    example_questions = [
        ("Minimum attendance", "What is the minimum attendance requirement?"),
        ("Re-examination rules", "What are the rules for applying for a re-examination?"),
        ("Leave of absence", "How can I apply for a leave of absence?"),
        ("Transcript procedure", "What is the procedure for obtaining a transcript?"),
        ("Failed a subject", "What happens if I fail a subject?")
    ]
    
    for i, (label, question) in enumerate(example_questions):
        with example_cols[i % len(example_cols)]:
            if st.button(label, key=f"ex_btn_{i}", help=question, use_container_width=True):
                st.session_state.pending_query = question

    st.divider()

    # Clear chat control
    col_chat_title, col_chat_clear = st.columns([5, 2])
    with col_chat_title:
        st.markdown("##### 🗨️ Your conversation")
    with col_chat_clear:
        if st.button("🗑️ Clear", use_container_width=True, help="Remove all messages from this conversation"):
            st.session_state.messages = []
            st.rerun()

    # Display Chat History
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                """
                <div style="text-align: center; padding: 36px 20px; color: #a8b6c9; background: #10223a; border: 1px solid #29415e; border-radius: 14px;">
                    <div style="font-size: 40px; margin-bottom: 10px;">📖</div>
                <div style="font-size: 16px; font-weight: 600; color: #edf2f8;">No messages yet</div>
                    <div style="font-size: 13px;">Ask a question above or type your academic policy inquiry below.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🎓"):
                    st.markdown(msg["content"])
                    
                    # If this message has citations, display them in an expandable card
                    if msg.get("citations"):
                        with st.expander(f"📚 View Document Provenance & Citations ({len(msg['citations'])} sources)"):
                            for c_idx, cite in enumerate(msg["citations"]):
                                st.markdown(
                                    f"""
                                    <div style="background-color: #122743; border: 1px solid #29415e; border-left: 3px solid #d7ad55; padding: 10px 12px; margin-bottom: 8px; border-radius: 8px;">
                                        <div style="margin-bottom: 4px;">
                                            <span class="citation-badge">📄 {cite['source']}</span>
                                            <span class="citation-badge">📍 Page {cite['page_number']}</span>
                                            <span class="citation-badge">🎯 Confidence {cite['confidence_pct']}%</span>
                                        </div>
                                        <div style="font-size: 12px; color: #c3cedb; font-style: italic;">
                                            "{cite['excerpt']}"
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    if msg.get("latency_ms"):
                        st.markdown(
                            f"<div class='chat-timestamp'>⚡ Response generated in {msg['latency_ms']} ms</div>",
                            unsafe_allow_html=True
                        )

    # Handle incoming query (from input box or clicked button)
    user_input = st.chat_input("Ask about attendance, re-exams, grading, transcripts, fees, leave...")
    active_query = st.session_state.pending_query or user_input

    if active_query:
        st.session_state.pending_query = None
        
        # Append User Message
        st.session_state.messages.append({
            "role": "user",
            "content": active_query,
            "citations": []
        })

        # Run RAG Query
        with st.spinner("Searching official college documents..."):
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            response = pipeline.query(active_query, conversation_history=history)

        # Append Assistant Message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "citations": response.get("citations", []),
            "latency_ms": response.get("latency_ms", 0)
        })

        st.rerun()


# ==================== TAB 2: DOCUMENT MANAGEMENT ====================
with tab_admin:
    st.markdown("### 📁 Admin Document Center & Knowledge Base")
    st.markdown("Upload official PDF handbooks, examination policies, calendar schedules, or attendance regulations.")

    col_upload, col_stats = st.columns([6, 4])
    
    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload Official College Regulations (PDF format only)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Extracts text per page, breaks into overlapping semantic chunks, and creates vector embeddings."
        )

        if uploaded_files:
            if st.button("🚀 Process & Index Uploaded Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_files = len(uploaded_files)
                new_chunks = []

                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Extracting and chunking {uploaded_file.name} ({idx+1}/{total_files})...")
                    try:
                        file_bytes = uploaded_file.read()
                        chunks = process_document(io.BytesIO(file_bytes), filename=uploaded_file.name)
                        new_chunks.extend(chunks)
                        
                        # Record in metadata table
                        st.session_state.uploaded_docs_meta.append({
                            "filename": uploaded_file.name,
                            "pages": max([c.page_number for c in chunks]) if chunks else 1,
                            "chunks": len(chunks),
                            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                        })
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                    
                    progress_bar.progress((idx + 1) / total_files)

                if new_chunks:
                    status_text.text("Embedding chunks into FAISS vector space...")
                    total_indexed = pipeline.build_index(new_chunks, append=True)
                    progress_bar.progress(1.0)
                    st.success(f"Successfully processed and indexed {len(new_chunks)} chunks! Total index size: {total_indexed} chunks.")
                    time.sleep(1)
                    st.rerun()

    with col_stats:
        st.markdown("##### 📊 Vector Knowledge Stats")
        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom: 12px;">
                <div class="metric-value">{len(pipeline.chunks)}</div>
                <div class="metric-label">Active Vector Chunks</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.uploaded_docs_meta)}</div>
                <div class="metric-label">Processed PDF Documents</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("##### 📚 Document Library")
    if st.session_state.uploaded_docs_meta:
        st.dataframe(
            st.session_state.uploaded_docs_meta,
            column_config={
                "filename": "Document Name",
                "pages": "Total Pages",
                "chunks": "Indexed Chunks",
                "added_at": "Indexed Date/Time"
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No documents uploaded yet. Upload a PDF above or click 'Load Sample' in the sidebar.")


# ==================== TAB 3: EVALUATION BENCHMARK ====================
with tab_eval:
    st.markdown("### 📊 Live RAG Evaluation Benchmark Dashboard")
    st.markdown(
        "Runs standardized queries through the active retrieval pipeline to empirically verify "
        "retrieval accuracy, source citations, response latency, and refusal on unsupported out-of-domain questions."
    )

    if st.button("🧪 Run Full Benchmark Evaluation Suite", type="primary"):
        with st.spinner("Evaluating benchmark queries across vector store..."):
            st.session_state.eval_results = run_evaluation_suite(pipeline)

    if st.session_state.eval_results:
        summary = st.session_state.eval_results["summary"]
        details = st.session_state.eval_results["details"]

        # 4 Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['retrieval_success_rate']}%</div>
                    <div class="metric-label">Retrieval Accuracy</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['citation_correctness_rate']}%</div>
                    <div class="metric-label">Citation Precision</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['refusal_accuracy_rate']}%</div>
                    <div class="metric-label">Refusal Precision</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with m_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['avg_latency_ms']} ms</div>
                    <div class="metric-label">Average Response Latency</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📋 Benchmark Test Execution Log")
        
        # Display individual test query results
        st.dataframe(
            details,
            column_config={
                "id": "Test ID",
                "category": "Query Category",
                "question": "Evaluated Query",
                "expected_page": "Expected Page",
                "retrieved_pages": "Retrieved Pages",
                "status": "Pipeline Status",
                "latency_ms": "Latency (ms)",
                "verdict": "Test Result"
            },
            use_container_width=True,
            hide_index=True
        )

        # Metric Calculation Glossary
        with st.expander("📖 How Each Evaluation Metric is Computed"):
            st.markdown(
                """
                - **Retrieval Accuracy Rate**: Percentage of in-domain questions where at least one retrieved chunk matches the verified policy page in the ground truth dataset:
                  $$\\text{Retrieval Rate} = \\left(\\frac{\\text{Correct Page Top-K Matches}}{\\text{Total In-Domain Questions}}\\right) \\times 100$$
                - **Citation Precision Rate**: Percentage of answered queries whose citations include validated source filenames and page numbers without hallucination.
                - **Refusal Precision**: Percentage of out-of-domain / unsupported queries where the system correctly returns the mandated policy refusal:
                  *\"I could not find sufficient information in the uploaded college documents to answer this question.\"*
                - **Response Latency**: Measured in milliseconds covering query embedding, FAISS similarity search, and answer synthesis.
                """
            )
    else:
        st.info("Click 'Run Full Benchmark Evaluation Suite' above to run the 10-query test suite and compute performance metrics.")


# ==================== TAB 4: SYSTEM GUIDE & DISCLAIMER ====================
with tab_guide:
    st.markdown("### 🏛️ System Architecture & Responsible AI Standards")
    
    st.markdown(
        """
        #### 1. RAG Workflow Pipeline
        ```mermaid
        graph LR
            A[Uploaded PDF Documents] --> B[pypdf Page Extractor]
            B --> C[Recursive Semantic Chunker]
            C --> D[Sentence Transformers Embedding]
            D --> E[(FAISS Vector Index)]
            F[Student Question] --> G[Query Embedder]
            G --> H[Cosine Similarity Search]
            E --> H
            H -->|Relevance Threshold Filter| I[Grounding System Prompt]
            I --> J[Grounded Generation Engine]
            J --> K[Answer + Verifiable Page Citations]
        ```
        """
    )
    
    st.divider()

    st.markdown("#### 2. Responsible AI Guidelines Implemented")
    st.markdown(
        """
        - **Zero-Hallucination Guardrails:** The model is prohibited from guessing, inferring outside policies, or estimating dates/fees.
        - **Definitive Refusal Policy:** Any question lacking sufficient evidentiary support in the vector store triggers the standardized refusal response.
        - **Auditability via Page Citations:** Every claim displays the source PDF filename, page number, and similarity confidence score.
        - **Conflicting Rules Detection:** When documents contain contradictory rules, both variants are reported for administrative review.
        """
    )

    st.divider()

    st.markdown("#### 3. LLM Configuration Options")
    st.markdown(
        """
        To use an external generative LLM, create a `.env` file in the project directory:
        ```env
        OPENAI_API_KEY=your_key_here
        OPENAI_MODEL_NAME=gpt-4o-mini
        ```
        *Compatible with OpenAI, Groq (`llama-3.3-70b-versatile`), Ollama, OpenRouter, and Together AI.*
        """
    )

    st.markdown(
        f"""
        <div class="disclaimer-box">
            <b>Official Institutional Disclaimer:</b> {OFFICIAL_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True
    )
