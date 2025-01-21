# **Web PDF Agent**

A powerful AI-driven assistant for querying and extracting information from websites and PDF documents. Built using LangChain, Cohere, and Playwright, this application processes web and PDF data sources to answer user questions interactively.

## **Features**

- **Website Integration:** Process and query information from any publicly available website.
- **PDF Integration:** Upload and process PDFs for real-time query and response generation.
- **Conversational Q&A:** Engage in a back-and-forth conversation to retrieve and refine answers based on the data.
- **Retrieval-Augmented Generation (RAG):** Combines document retrieval with AI to generate contextually relevant answers.

## **Technologies Used**

- **Streamlit:** For building the interactive web UI.
- **LangChain:** For document processing, text splitting, and conversational chains.
- **Cohere:** For AI-powered language models for natural language understanding.
- **Playwright:** For web scraping and data extraction from websites.
- **Chroma:** For efficient vector storage and retrieval.
- **NLTK:** For natural language processing tasks.
- **PyMuPDF (fitz):** For handling and processing PDF documents.

## **How It Works**

This app allows users to interact with their data in an intuitive way by providing two main options:
1. **Website:** Enter a website URL, and the app scrapes and processes the site’s content for Q&A.
2. **PDF:** Upload a PDF document, which the app processes to allow querying its contents.

Once a data source is processed, users can ask questions, and the app will use the relevant data to generate accurate answers.

## **Installation**

### **Clone the Repository**

Clone this repository to your local machine:

```bash
git clone https://github.com/sk-tech24/WebPdfAgent.git
cd WebPdfAgent
```

### **Dependencies**

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### **Set Up Environment Variables**

Make sure to set up necessary environment variables like the **Cohere API Key** for NLP tasks. You can add them to a `.env` file:

```env
COHERE_API_KEY=your_cohere_api_key
```

### **Running the App Locally**

To run the app locally, use Streamlit:

```bash
streamlit run app.py
```

## **Deploy on Streamlit Cloud**

1. Push the code to your GitHub repository.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Click "New app" and link it to your repository.
4. Set the correct environment variables if necessary.

## **Usage**

Once deployed, simply access the app via your browser. You can either:
1. **Enter a Website URL** to process data from a webpage.
2. **Upload a PDF** to interact with its content.

Ask questions related to the data source, and the AI will provide contextually accurate answers based on the uploaded or scraped content.

## **Contributing**

If you'd like to contribute to this project, feel free to fork the repository, create a new branch, and submit a pull request.

Please ensure that you follow the existing code style and write tests for new functionality.

## **License**

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## **Acknowledgements**

- **LangChain:** For integrating AI-driven document processing and conversation management.
- **Cohere:** For the powerful language models.
- **Streamlit:** For building interactive web applications with ease.

---
