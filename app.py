import streamlit as st
from RagWebsiteAgent import RAGWebsiteAgent
from PdfQaAgent import PdfQaAgent
import os
from urllib.parse import urlparse

# Ensure Playwright is installed
# os.system("playwright install")

import nltk
nltk.data.path.append("./nltk_data/")
nltk.download("punkt", download_dir="./nltk_data/")
nltk.download("punkt_tab", download_dir="./nltk_data/")
nltk.download("averaged_perceptron_tagger_eng", download_dir="./nltk_data/")

def is_valid_url(url):
    """Check if the provided URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def handle_user_input(question, agent, spinner_placeholder):
    """Handle user input and update chat history."""
    if st.session_state.is_website_processed or st.session_state.is_pdf_processed:
        with spinner_placeholder.container():
            with st.spinner("Generating answer..."):
                try:
                    response = agent.ask_question(question)
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error("Failed to process your query. Please try again.")
    else:
        st.error("Please process the data source first.")

def reset_chat():
    st.session_state.chat_history = []
    st.session_state.conversation = None
    st.session_state.is_website_processed = False
    st.session_state.is_pdf_processed = False
    st.session_state.current_source = None
    st.session_state.rag_agent = None
    st.session_state.pdf_agent = None

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Chat with Data",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="auto"
    )
    st.header("Chat with Your Data 🤖")

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "is_website_processed" not in st.session_state:
        st.session_state.is_website_processed = False
    if "is_pdf_processed" not in st.session_state:
        st.session_state.is_pdf_processed = False
    if "current_source" not in st.session_state:
        st.session_state.current_source = None
    if "rag_agent" not in st.session_state:
        st.session_state.rag_agent = None
    if "pdf_agent" not in st.session_state:
        st.session_state.pdf_agent = None

    # Sidebar for source selection
    st.sidebar.title("Select Data Source")
    source_type = st.sidebar.radio(
        "Choose a data source:", 
        ("Website", "PDF"), 
        key='data_source', 
        on_change=reset_chat
    )

    if source_type == "Website":
        st.session_state.source_type = "website"
        with st.sidebar:
            st.subheader("Process a Website:")
            base_url = st.text_input("Enter website URL:", key="website_url_input")

            if base_url and not st.session_state.is_website_processed:
                if is_valid_url(base_url):
                    with st.spinner("Processing the website..."):
                        try:
                            st.session_state.rag_agent = RAGWebsiteAgent(base_url=base_url)
                            st.session_state.rag_agent.prepare_qa_system()
                            st.session_state.current_source = base_url
                            st.session_state.conversation = True
                            st.session_state.is_website_processed = True
                            st.success("Website data processed. You can start asking questions!")
                        except Exception as e:
                            st.error(f"Error processing website: {e}")
                else:
                    st.error("Please enter a valid website URL.")

    elif source_type == "PDF":
        st.session_state.source_type = "pdf"
        with st.sidebar:
            st.subheader("Upload a PDF:")
            pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

            if pdf_file and not st.session_state.is_pdf_processed:
                with st.spinner("Processing the PDF..."):
                    try:
                        pdf_path = os.path.join("uploads", pdf_file.name)
                        os.makedirs("uploads", exist_ok=True)
                        with open(pdf_path, "wb") as f:
                            f.write(pdf_file.getbuffer())

                        st.session_state.pdf_agent = PdfQaAgent(pdf_path=pdf_path)
                        st.session_state.pdf_agent.prepare_qa_system()
                        st.session_state.current_source = pdf_file.name
                        st.session_state.conversation = True
                        st.session_state.is_pdf_processed = True
                        st.success("PDF data processed. You can start asking questions!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")

    if st.session_state.current_source:
        st.subheader(f"Interacting with: {st.session_state.current_source}")
    else:
        st.subheader("No data source processed yet.")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    input_container = st.container()
    with input_container:
        if st.session_state.conversation:
            def handle_and_clear_input():
                """Handle user input and clear the text input field."""
                handle_user_input(
                    st.session_state.user_input,
                    st.session_state.rag_agent if st.session_state.source_type == "website" else st.session_state.pdf_agent,
                    spinner_placeholder,
                )
                st.session_state.user_input = ""  # Clear the input text box

            st.text_input(
                "Ask anything about your data:",
                key="user_input",
                on_change=handle_and_clear_input,
            )
            spinner_placeholder = st.empty()
        else:
            st.info("Please select a data source and wait until it's ready to ask questions.")

if __name__ == "__main__":
    main()
