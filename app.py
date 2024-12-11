import streamlit as st
import asyncio
import logging
from backend import prepare_rag_llm, generate_answer, embedding_storing, split_doc_with_metadata, read_file
from pypdf import PdfReader

# Set up logging
logging.basicConfig(level=logging.INFO)  # Adjust level as needed

# Streamlit application setup
async def main():
    st.set_page_config(layout="wide")
    st.title("Financial Analysis Chatbot")

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Streamlit sidebar for file upload and chatbot creation
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Upload PDF, Text, or CSV file", type=["pdf", "txt", "csv"])
        
        if uploaded_file is not None:
            # Read uploaded file asynchronously
            document = await read_file(uploaded_file, uploaded_file.type)
            
            # Display document content
            st.sidebar.subheader("Uploaded Document Preview")
            st.sidebar.text(document[:500])  # Display first 500 characters
            
            # Create chatbot button
            if st.sidebar.button("Create Chatbot"):
                # Parameters for document processing and RAG
                chunk_size = 500  # Hardcoded parameter
                chunk_overlap = 50  # Hardcoded parameter
                token = "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"  # Hardcoded token
                llm_model = "mistralai/Mistral-7B-Instruct-v0.2"  # Hardcoded model
                instruct_embeddings = "all-MiniLM-L6-v2"  # Hardcoded embeddings
                temperature = 1.0  # Hardcoded temperature
                max_length = 1000  # Hardcoded max length

                # Document file name
                file_name = uploaded_file.name

                # Split document into chunks
                split_document = split_doc_with_metadata(document, chunk_size, chunk_overlap, file_name)
                
                # Store document embeddings with file_name
                vector_store_path = await embedding_storing(
                    llm_model, instruct_embeddings, document, chunk_size, chunk_overlap, file_name
                )
                
                # Prepare RAG model asynchronously
                st.session_state.conversation = await prepare_rag_llm(
                    token, llm_model, instruct_embeddings, vector_store_path, temperature, max_length
                )
                
                st.success("Chatbot created successfully!")

    # Display chat history and handle new messages
    if "conversation" in st.session_state:
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input and handle responses
        user_input = st.chat_input("Ask a question")
        if user_input:
            st.session_state.history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Generate and display the answer asynchronously
            async def handle_question():
                try:
                    answer, citations = await generate_answer(user_input, st.session_state.conversation)
                    st.session_state.history.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.write(answer)
                        if citations:
                            st.write("Citations:")
                            for citation in citations:
                                st.write(citation)
                except Exception as e:
                    st.error(f"An error occurred while processing the question: {str(e)}")
                    logging.error(f"Error generating answer: {str(e)}")
            
            # Run the async task
            asyncio.create_task(handle_question())

# Run the Streamlit application
if __name__ == "__main__":
    asyncio.run(main())
