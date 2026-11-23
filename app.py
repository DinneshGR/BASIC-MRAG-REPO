
import os
import streamlit as st

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from google import genai

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")


# ==========================================
# 2. INITIALIZE GEMINI CLIENT
# ==========================================

client = genai.Client(api_key=GEMINI_API_KEY)

EMBEDDING_MODEL = "gemini-embedding-2"
LLM_MODEL = "gemini-3.6-flash"


# ==========================================
# 3. CUSTOM GEMINI EMBEDDING CLASS
# ==========================================


class GeminiEmbeddings(Embeddings):

    def __init__(self):
        self.client = client
        self.model = "gemini-embedding-2"

    def embed_documents(self, texts):

        embeddings = []

        for text in texts:

            response = self.client.models.embed_content(
                model=self.model,
                contents=text
            )

            embedding = response.embeddings[0].values

            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text):

        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )

        return response.embeddings[0].values

# ==========================================
# 4. EXTRACT TEXT FROM MULTIPLE PDF FILES
# ==========================================

def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        pdf_reader = PdfReader(pdf)

        for page_num, page in enumerate(pdf_reader.pages):

            extracted_text = page.extract_text()

            if extracted_text:
                text += extracted_text + "\n"

    return text


# ==========================================
# 5. SPLIT TEXT INTO CHUNKS
# ==========================================

def get_text_chunks(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks


# ==========================================
# 6. CREATE FAISS VECTOR DATABASE
# ==========================================

def get_vectorstore(text_chunks):

    embeddings = GeminiEmbeddings()

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore

# ==========================================
# 7. RETRIEVE RELEVANT DOCUMENTS
# ==========================================

def get_relevant_documents(vectorstore, question):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    documents = retriever.invoke(question)

    return documents


# ==========================================
# 8. GEMINI RESPONSE GENERATION
# ==========================================

def generate_response(question, documents, chat_history):

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    history = ""

    for message in chat_history[-6:]:

        if isinstance(message, HumanMessage):
            history += f"User: {message.content}\n"

        elif isinstance(message, AIMessage):
            history += f"Assistant: {message.content}\n"

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using the provided document context.

Rules:
1. Use the supplied context as your primary source.
2. If the answer is not available in the context, say so.
3. Do not invent facts.
4. Give clear and structured answers.
5. Use conversation history to understand follow-up questions.

CONVERSATION HISTORY:
{history}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text


# ==========================================
# 9. STREAMLIT MAIN APPLICATION
# ==========================================

def main():

    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Chat with Multiple PDFs")

    st.caption(
        "Ask questions about your documents using Gemini and FAISS."
    )

    # Initialize session state

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "processed" not in st.session_state:
        st.session_state.processed = False

    # ======================================
    # SIDEBAR
    # ======================================

    with st.sidebar:

        st.header("📂 Your Documents")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process Documents", use_container_width=True):

            if not pdf_docs:

                st.warning("Please upload at least one PDF.")

            else:

                with st.spinner("Processing your PDFs..."):

                    try:

                        # Step 1: Extract text
                        raw_text = get_pdf_text(pdf_docs)

                        if not raw_text.strip():
                            st.error(
                                "No extractable text found. "
                                "Your PDFs may be scanned documents."
                            )
                            st.stop()

                        # Step 2: Chunk text
                        text_chunks = get_text_chunks(raw_text)

                        if not text_chunks:
                            st.error("No text chunks were generated.")
                            st.stop()

                        # Step 3: Create vector database
                        vectorstore = get_vectorstore(text_chunks)

                        # Step 4: Store in session
                        st.session_state.vectorstore = vectorstore
                        st.session_state.chat_history = []
                        st.session_state.processed = True

                        st.success("Documents processed successfully!")

                        st.write(f"Total chunks: {len(text_chunks)}")

                    except Exception as e:

                        st.error(f"Processing failed: {str(e)}")

        if st.button("Clear Chat", use_container_width=True):

            st.session_state.chat_history = []
            st.rerun()

    # ======================================
    # CHAT INTERFACE
    # ======================================

    if st.session_state.processed:

        for message in st.session_state.chat_history:

            if isinstance(message, HumanMessage):

                with st.chat_message("user"):
                    st.markdown(message.content)

            elif isinstance(message, AIMessage):

                with st.chat_message("assistant"):
                    st.markdown(message.content)

        user_question = st.chat_input(
            "Ask a question about your PDFs..."
        )

        if user_question:

            # Display user question
            with st.chat_message("user"):
                st.markdown(user_question)

            st.session_state.chat_history.append(
                HumanMessage(content=user_question)
            )

            with st.chat_message("assistant"):

                with st.spinner("Searching documents..."):

                    try:

                        # Step 1: Retrieve relevant chunks
                        documents = get_relevant_documents(
                            st.session_state.vectorstore,
                            user_question
                        )

                        # Step 2: Generate answer
                        answer = generate_response(
                            user_question,
                            documents,
                            st.session_state.chat_history
                        )

                        st.markdown(answer)

                        # Display sources
                        with st.expander("View retrieved sources"):

                            for i, document in enumerate(documents):

                                st.markdown(
                                    f"**Retrieved Chunk {i + 1}**"
                                )

                                st.write(document.page_content)

                        # Save assistant response
                        st.session_state.chat_history.append(
                            AIMessage(content=answer)
                        )

                    except Exception as e:

                        st.error(f"Error generating response: {str(e)}")

    else:

        st.info(
            "Upload your PDF documents in the sidebar "
            "and click Process Documents to start chatting."
        )


# ==========================================
# 10. APPLICATION ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
