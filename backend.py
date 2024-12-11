import asyncio
import os
import tempfile
import pandas as pd
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import functools
from langchain.chains import ConversationalRetrievalChain

# Function to read PDF files asynchronously
async def read_pdf(file):
    document = ""
    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()
    return document

# Function to read text files asynchronously
async def read_txt(file):
    document = str(file.getvalue())
    document = document.replace("\\n", " \\n ").replace("\\r", " \\r ")
    return document

# Function to read CSV files asynchronously
async def read_csv(file):
    df = pd.read_csv(file)
    document = df.to_string(index=False)  # Convert dataframe to string
    return document

# Function to handle different file types
async def read_file(file, file_type):
    if file_type == "application/pdf":
        return await read_pdf(file)
    elif file_type == "text/plain":
        return await read_txt(file)
    elif file_type == "text/csv":
        return await read_csv(file)

def split_doc_with_metadata(document, chunk_size, chunk_overlap, file_name):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)
    
    # Add metadata to each document chunk
    for i, doc in enumerate(split):
        doc.metadata = {
            "file_name": file_name,
            "chunk_number": i + 1
        }
    
    return split

async def embedding_storing(model_name, instruct_embeddings, document, chunk_size, chunk_overlap, file_name):
    # Split document into chunks with metadata
    split = split_doc_with_metadata(document, chunk_size, chunk_overlap, file_name)
    
    # Load embeddings instructor
    instructor_embeddings = HuggingFaceInstructEmbeddings(
        model_name=instruct_embeddings, model_kwargs={"device": "cpu"}
    )

    # Create vector store
    db = FAISS.from_documents(split, instructor_embeddings)
    
    # Save vector store
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        vector_store_path = f"vector_store/{os.path.basename(temp_file.name)}.faiss"
        db.save_local(vector_store_path)
    
    return vector_store_path

async def prepare_rag_llm(token, llm_model, instruct_embeddings, vector_store_path, temperature, max_length):
    # Load embeddings instructor
    instructor_embeddings = HuggingFaceInstructEmbeddings(
        model_name=instruct_embeddings, model_kwargs={"device": "cpu"}
    )

    # Load vector store
    loaded_db = FAISS.load_local(
        vector_store_path, instructor_embeddings, allow_dangerous_deserialization=True
    )

    # Load LLM model
    llm = HuggingFaceHub(
        repo_id=llm_model,
        model_kwargs={"temperature": temperature, "max_length": max_length},
        huggingfacehub_api_token=token
    )

    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Create the chatbot asynchronously
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(),
        return_source_documents=True,
        memory=memory,
    )

    return qa_conversation

async def generate_answer(question, conversation):
    answer = "An error occurred while generating the answer."
    citations = []
    try:
        response = conversation({"question": question})
        if response:
            answer = response.get("answer", "").split("Helpful Answer:")[-1].strip()
            # Extract citations with document name and chunk number
            source_docs = response.get("source_documents", [])
            citations = [
                f"Source: {doc.metadata.get('file_name', 'Unknown file')} - Chunk {doc.metadata.get('chunk_number', 'Unknown chunk')}"
                for doc in source_docs
            ]
    except Exception as e:
        answer = f"Error: {str(e)}"
    
    return answer, citations

# Caching embedding storing function
@functools.lru_cache(maxsize=128)
async def cached_embedding_storing(model_name, document, chunk_size, chunk_overlap):
    return await embedding_storing(model_name, document, chunk_size, chunk_overlap)
