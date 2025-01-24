import os
import pickle
import logging
from urllib.parse import urljoin, urlparse
# from playwright.sync_api import sync_playwright
import sys
# import shutil
import requests

from langchain.prompts import PromptTemplate
# from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_cohere import ChatCohere
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from crawler import WebCrawler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Custom prompt template for the QA system
custom_prompt_template = """
You are a professional and intelligent assistant. Your task is to answer questions based on the provided context. Always be polite and respectful, no matter the nature of the question. Respond in a concise and structured manner, making it easy for the user to extract relevant information. If the context does not contain sufficient information to answer the question, kindly inform the user in a clear and respectful way that you cannot provide an answer based on the current context. Avoid fabricating information and always prioritize clarity and brevity.

Context:
{context}

Question:
{question}

Answer:
- Keep the answer short and to the point.
- Use bullet points, lists, or short paragraphs whenever needed.
- Highlight important details or steps clearly.
- If unsure, politely mention that more information is needed.
"""

class RAGWebsiteAgent:
    def __init__(self, base_url, persist_directory="chroma_store", environment="development"):
        if not base_url:
            raise ValueError("Base URL cannot be empty.")
        if not urlparse(base_url).scheme.startswith("http"):
            raise ValueError(f"Invalid URL: {base_url}. Please provide a valid HTTP/HTTPS URL.")

        self.base_url = base_url.rstrip("/")
        self.domain_name = urlparse(base_url).netloc.replace('.', '_')
        self.persist_directory = os.path.join(persist_directory, self.domain_name)
        self.environment = environment
        self.visited_urls = set()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None

        # Initialize memory for conversation history
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key="answer", return_messages=True)

        # Read Cohere API Key from environment variable
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set. Please set the API key and try again.")

    def is_url_reachable(self, url):
        """Check if the URL is reachable."""
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return True
        except requests.HTTPError as e:
            logger.error(f"HTTP error while reaching URL {url}: {e}")
        except requests.ConnectionError as e:
            logger.error(f"Connection error while reaching URL {url}: {e}")
        except requests.Timeout:
            logger.error(f"Timeout while reaching URL {url}")
        except requests.RequestException as e:
            logger.error(f"Error while reaching URL {url}: {e}")
        return False

    def prepare_qa_system(self):
        """Prepare the QA system by crawling and indexing the website content."""
        if os.path.exists(self.persist_directory):
            logger.info(f"Loading existing Chroma vector store for {self.base_url}...")
            try:
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
            except Exception as e:
                logger.error(f"Failed to load existing vector store: {e}")
                raise
        else:
            if not self.is_url_reachable(self.base_url):
                raise ValueError(f"The URL {self.base_url} is not reachable. Please enter a valid URL.")

            logger.info(f"Crawling the website {self.base_url} and creating a new vector store...")
            # documents = self.crawl_website(self.base_url)
            crawler = WebCrawler(base_url=self.base_url, throttle_delay=2, use_dynamic=True)
            # Start crawling
            crawler.crawl()
            # Get the extracted documents
            documents = crawler.get_documents()
            if not documents:
                raise ValueError(f"No documents were loaded from the URL {self.base_url}. Please check the URL and try again.")

            texts, metadata = [], []
            for doc in documents:
                try:
                    splits = self.text_splitter.split_text(doc.page_content)
                    texts.extend(splits)
                    metadata.extend([doc.metadata] * len(splits))
                except Exception as e:
                    logger.warning(f"Error while splitting document: {e}")

            try:
                os.makedirs(self.persist_directory, exist_ok=True)
                with open(os.path.join(self.persist_directory, "documents.pkl"), "wb") as f:
                    pickle.dump(documents, f)

                self.vector_store = Chroma.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadata,
                    persist_directory=self.persist_directory,
                )
                self.vector_store.persist()
            except Exception as e:
                logger.error(f"Failed to create vector store: {e}")
                raise

        retriever = self.vector_store.as_retriever()
        retriever.search_kwargs["k"] = 5

        llm = ChatCohere(model="command-r-plus-08-2024", cohere_api_key=self.cohere_api_key)
        
        custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=custom_prompt_template,
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
        )

    # def extract_dynamic_urls(self, url, collected_urls):
    #     """Extract dynamic URLs from the given page using Selenium."""
    #     chrome_options = Options()
    #     chrome_options.add_argument("--headless")
    #     chrome_options.add_argument("--disable-gpu")
    #     chrome_options.add_argument("--no-sandbox")
    #     chrome_options.add_argument("--disable-dev-shm-usage")

    #     service = Service(ChromeDriverManager().install())
    #     driver = webdriver.Chrome(service=service, options=chrome_options)

    #     try:
    #         driver.get(url)
    #         driver.implicitly_wait(10)  # Wait for elements to load
    #         links = driver.find_elements(By.XPATH, "//a[@href]")
    #         urls = [urljoin(url, link.get_attribute("href")) for link in links]
    #         collected_urls.update(set(urls))
    #     except Exception as e:
    #         logger.error(f"Error navigating to {url}: {e}")
    #     finally:
    #         driver.quit()

    # def crawl_website(self, start_url):
    #     """Crawl the website and extract content using Selenium."""
    #     urls_to_scrape = {start_url}
    #     collected_urls = set()

    #     max_urls = None if self.environment == "production" else 10

    #     while urls_to_scrape and (max_urls is None or len(collected_urls) < max_urls):
    #         url = urls_to_scrape.pop()
    #         logger.info(f"Crawling URL: {url}")
    #         self.extract_dynamic_urls(url, collected_urls)
    #         urls_to_scrape.update(collected_urls - self.visited_urls)
    #         self.visited_urls.update(collected_urls)

    #         if max_urls is not None and len(collected_urls) >= max_urls:
    #             break

    #     logger.info(f"Collected {len(collected_urls)} unique URLs")

    #     clean_urls = [u for u in collected_urls if urlparse(u).scheme in ("http", "https")]
    #     loader = PlaywrightURLLoader(clean_urls, continue_on_failure=True)  # Keep using PlaywrightURLLoader for loading documents
    #     documents = loader.load()

    #     logger.info(f"Number of documents loaded: {len(documents)}")
    #     return documents

    def ask_question(self, query):
        """Ask a question and retrieve the answer using the QA system."""
        if not hasattr(self, "qa_chain"):
            raise ValueError("QA system not prepared. Call `prepare_qa_system()` first.")

        try:
            response = self.qa_chain({"question": query})
            return response.get("answer", "Sorry, no answer found.")
        except Exception as e:
            logger.error(f"Error while processing the question: {e}")
            return "An error occurred while processing your question. Please try again later."

    def get_chat_history(self):
        """Return the chat history."""
        try:
            return self.memory.load_memory_variables({})["chat_history"]
        except Exception as e:
            logger.error(f"Error fetching chat history: {e}")
            return "Could not retrieve chat history."
