import os
import pickle
import logging
import fitz  # PyMuPDF
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_cohere import ChatCohere

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PdfQaAgent:
    def __init__(self, pdf_path, persist_directory="pdf_store"):
        if not pdf_path or not os.path.isfile(pdf_path):
            raise ValueError(f"Invalid PDF path: {pdf_path}. Please provide a valid file path.")
        if not pdf_path.endswith(".pdf"):
            raise ValueError(f"File {pdf_path} is not a valid PDF. Please provide a PDF file.")

        self.pdf_path = pdf_path
        self.persist_directory = os.path.join(persist_directory, os.path.splitext(os.path.basename(pdf_path))[0])
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None

        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)

        # Read Cohere API Key from environment variable
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set. Please configure it and try again.")

    def extract_text_from_pdf(self):
        """Extract text from the PDF file."""
        try:
            doc = fitz.open(self.pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            if not text.strip():
                raise ValueError(f"No extractable text found in the PDF: {self.pdf_path}.")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise ValueError(f"Failed to extract text from the PDF: {e}")

    def prepare_qa_system(self):
        """Prepare the QA system by embedding and indexing the PDF content."""
        try:
            if os.path.exists(self.persist_directory):
                logger.info(f"Loading existing Chroma vector store for {self.pdf_path}...")
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
            else:
                logger.info(f"Processing the PDF {self.pdf_path} and creating a new vector store...")
                text = self.extract_text_from_pdf()
                splits = self.text_splitter.split_text(text)
                if not splits:
                    raise ValueError(f"Failed to split the text from {self.pdf_path} into chunks.")
                metadata = [{"source": self.pdf_path}] * len(splits)

                os.makedirs(self.persist_directory, exist_ok=True)
                with open(os.path.join(self.persist_directory, "documents.pkl"), "wb") as f:
                    pickle.dump(splits, f)

                self.vector_store = Chroma.from_texts(
                    texts=splits,
                    embedding=self.embeddings,
                    metadatas=metadata,
                    persist_directory=self.persist_directory,
                )
                self.vector_store.persist()

            retriever = self.vector_store.as_retriever()
            retriever.search_kwargs["k"] = 5  # Retrieve top 5 results

            llm = ChatCohere(model="command-r-plus-08-2024", cohere_api_key=self.cohere_api_key)

            # Define the conversational retrieval chain
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
            )
        except Exception as e:
            logger.error(f"Error preparing QA system: {e}")
            raise ValueError(f"Failed to prepare the QA system: {e}")

    def ask_question(self, query):
        """Ask a question and retrieve the answer using the QA system."""
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty. Please provide a valid question.")
            if not hasattr(self, "qa_chain"):
                raise ValueError("QA system not prepared. Call `prepare_qa_system()` first.")

            response = self.qa_chain({"question": query})
            answer = response.get("answer", "No answer found.")
            return answer
        except Exception as e:
            logger.error(f"Error during question answering: {e}")
            raise ValueError(f"Failed to process your query: {e}")

    def get_chat_history(self):
        """Return the chat history."""
        try:
            return self.memory.load_memory_variables({}).get("chat_history", [])
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
            raise ValueError(f"Failed to retrieve chat history: {e}")
