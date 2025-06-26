import os
import streamlit as st
from dotenv import load_dotenv
import tempfile

# Import libraries for RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import QdrantClient

#Caching Function for the Vector Store
@st.cache_resource
def create_vector_store(_uploaded_file):
    """
    Creates and caches a Qdrant vector store for a given uploaded PDF file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.info(f"Processing '{_uploaded_file.name}'... This may take a moment.")
    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Using an in-memory/local docker Qdrant instance
    qdrant = Qdrant.from_documents(
        all_splits,
        embeddings,
        location="http://localhost:6333",
        collection_name=f"filename_{_uploaded_file.file_id}",
    )
    
    os.remove(tmp_file_path) 
    st.success(f"'{_uploaded_file.name}' has been processed. You can now ask questions.")
    return qdrant


#Streamlit app setup

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("Google API Key not found. Please add it to your .env file.")
    st.stop()

st.set_page_config(page_title="Ask Your PDF")
st.title(" Ask Your PDF with Chunk Visualization")
st.write("Upload a PDF, see how it's chunked, and ask questions about its content.")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    try:
        # Create and cache the vector store from the uploaded file
        vector_store = create_vector_store(uploaded_file)

        # Chunk Visualization
        #with st.expander("Visualize Document Chunks"):
           # visualize_chunks(vector_store)
        
        # Set up the RAG chain for Q&A
        retriever = vector_store.as_retriever()
        template = """
        Use the following context from the document to answer the question. Do not make up answers. If the answer is not in the context, say "Irrelevant question, please ask another one".
        Context: {context}
        Question: {question}
        Answer:
        """
        prompt = PromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Ask a question
        question = st.text_input("Ask a question about the document:")
        if question:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(question)
                st.markdown("### Answer")
                st.write(response)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a PDF file to begin.")