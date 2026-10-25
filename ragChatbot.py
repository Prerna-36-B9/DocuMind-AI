import os
import hashlib
import streamlit as st
import pdfplumber

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0b1120 0%, #111827 55%, #1e1b4b 100%);
        color: #f8fafc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        padding: 28px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            rgba(99, 102, 241, 0.22),
            rgba(168, 85, 247, 0.12)
        );
        border: 1px solid rgba(167, 139, 250, 0.30);
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 8px;
        color: #ffffff;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .feature-card {
        padding: 18px;
        border-radius: 16px;
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.20);
        min-height: 120px;
    }

    .feature-card h3 {
        color: #c4b5fd;
        font-size: 1.05rem;
    }

    .feature-card p {
        color: #cbd5e1;
        font-size: 0.9rem;
    }

    .stButton > button {
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        font-weight: 600;
        padding: 0.55rem 1rem;
    }

    .stButton > button:hover {
        border: 1px solid #c4b5fd;
        color: white;
    }

    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #334155;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# =========================================================
# PDF PROCESSING
# =========================================================

def process_pdf(uploaded_file):

    pdf_bytes = uploaded_file.getvalue()

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Avoid processing the same PDF repeatedly.
    if (
        st.session_state.processed_file_hash == file_hash
        and st.session_state.vector_store is not None
    ):
        return True

    try:
        extracted_text = []

        with pdfplumber.open(uploaded_file) as pdf:

            for page_number, page in enumerate(pdf.pages, start=1):

                page_text = page.extract_text()

                if page_text:
                    extracted_text.append(
                        f"[Page {page_number}]\n{page_text}"
                    )

        full_text = "\n\n".join(extracted_text)

        if not full_text.strip():
            st.error(
                "No readable text was found in this PDF. "
                "It may be a scanned document."
            )
            return False

        # Split the document into manageable chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = text_splitter.split_text(full_text)

        if not chunks:
            st.error("No text chunks could be created.")
            return False

        # Generate embeddings and create the FAISS index.
        embeddings = load_embeddings()

        with st.spinner("Creating your document's vector database..."):

            vector_store = FAISS.from_texts(
                texts=chunks,
                embedding=embeddings
            )

        # Store the processed document information.
        st.session_state.vector_store = vector_store
        st.session_state.processed_file_hash = file_hash
        st.session_state.document_name = uploaded_file.name
        st.session_state.chunk_count = len(chunks)

        # Start a fresh conversation for a new document.
        st.session_state.messages = []

        return True

    except Exception as error:
        st.error(f"PDF processing failed: {error}")
        return False


# =========================================================
# ANSWER GENERATION
# =========================================================

def answer_question(question):

    vector_store = st.session_state.vector_store

    # Find the four most relevant sections of the PDF.
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return (
            "I couldn't find relevant information in your document."
        ), []

    # Combine retrieved document chunks.
    context = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    # Create the prompt.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are DocuMind AI, a helpful assistant that answers
            questions using the uploaded PDF document.

            Guidelines:
            1. Answer using only the provided document context.
            2. Explain answers clearly and thoroughly.
            3. Include relevant facts, numbers, and examples.
            4. Use bullet points when appropriate.
            5. Never invent information.
            6. If the answer is not present in the context,
               politely explain that the document does not
               provide enough information.
            7. Mention page numbers whenever available.

            Document context:
            {context}
            """
        ),
        (
            "human",
            "{question}"
        )
    ])

    # Use the free, locally running Ollama model.
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.3
    )

    # Generate the answer.
    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content, relevant_docs

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 📚 DocuMind AI")

    st.caption("Your personal document assistant")

    st.divider()

    st.markdown("### 📄 Your Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF to get started",
        type=["pdf"],
        help="Choose a PDF document you want to ask questions about."
    )

    if uploaded_file is not None:

        file_hash = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()

        is_new_file = (
            file_hash != st.session_state.processed_file_hash
        )

        if is_new_file:

            if st.button(
                "⚡ Process PDF",
                use_container_width=True
            ):

                with st.spinner(
                    "Reading PDF and generating embeddings..."
                ):

                    success = process_pdf(uploaded_file)

                if success:
                    st.success("Your document is ready!")
                    st.rerun()

        elif st.session_state.vector_store is not None:

            st.success("✅ Document processed")

    if st.session_state.vector_store is not None:

        st.divider()

        st.markdown("### 📊 Document Statistics")

        st.write(
            f"**File:** {st.session_state.document_name}"
        )

        st.metric(
            "Text chunks",
            st.session_state.chunk_count
        )

        st.caption("Embedding model: MiniLM")

        st.caption("Vector database: FAISS")

        if st.button(
            "🗑️ Clear conversation",
            use_container_width=True
        ):
            st.session_state.messages = []
            st.rerun()

    st.divider()

    st.markdown("### ⚙️ How it works")

    st.markdown("""
    1. Upload a PDF.
    2. Extract its text.
    3. Split text into chunks.
    4. Generate embeddings.
    5. Retrieve relevant chunks.
    6. Generate an AI answer.
    """)

    st.markdown(
        '<p class="small-note">Built with Streamlit, '
        'LangChain, Hugging Face and FAISS.</p>',
        unsafe_allow_html=True
    )


# =========================================================
# MAIN PAGE
# =========================================================

st.markdown("""
<div class="hero">
    <h1>✨ Chat with your Documents</h1>
    <p>
        Upload a PDF, ask questions, and explore your document
        with the help of AI.
    </p>
</div>
""", unsafe_allow_html=True)


# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📄 Smart Reading</h3>
        <p>Extract and process text from your PDF documents.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🔎 Context Search</h3>
        <p>Find document sections relevant to your question.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h3>🤖 AI Answers</h3>
        <p>Generate helpful answers grounded in your document.</p>
    </div>
    """, unsafe_allow_html=True)


st.divider()


# =========================================================
# CHAT SECTION
# =========================================================

st.markdown("## 💬 Ask your document")

if st.session_state.vector_store is None:

    st.info(
        "👈 Upload a PDF in the sidebar and click "
        "**Process PDF** to start chatting."
    )

else:

    st.caption(
        f"Currently chatting with: "
        f"{st.session_state.document_name}"
    )

    # Display existing conversation.
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show retrieved sources for assistant answers.
            if message["role"] == "assistant":
                sources = message.get("sources", [])

                if sources:
                    with st.expander("📚 View retrieved context"):

                        for index, doc in enumerate(sources, start=1):
                            st.markdown(
                                f"**Retrieved section {index}**"
                            )
                            st.write(doc.page_content)

    # Chat input.
    user_question = st.chat_input(
        "Ask anything about your PDF..."
    )

    if user_question:

        # Display and store the user's question.
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate the answer.
        with st.chat_message("assistant"):

            with st.spinner("Searching your document..."):

                try:

                    response, sources = answer_question(
                        user_question
                    )

                    st.markdown(response)

                    with st.expander("📚 View retrieved context"):

                        for index, doc in enumerate(sources, start=1):
                            st.markdown(
                                f"**Retrieved section {index}**"
                            )
                            st.write(doc.page_content)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

                except Exception as error:

                    st.error(f"Unable to generate an answer: {error}")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; padding:10px;">
        ✨ DocuMind AI · Built with ❤️ using Python
    </div>
    """,
    unsafe_allow_html=True
)
