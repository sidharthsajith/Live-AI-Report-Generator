This is a Streamlit application that allows users to upload documents, crawl websites, and ask questions to retrieve relevant information.  The application utilizes sentence embeddings and a Faiss index for efficient retrieval.

**Features:**

- Upload documents in various formats (PDF, DOCX, TXT, PPTX, XLSX, CSV, JSON)
- Crawl websites to extract text content
- Integrate with MongoDB and SQLite databases (optional)
- Process documents and build a knowledge base using sentence embeddings
- Ask questions and receive concise, factual answers based on the uploaded documents, crawled data, and connected databases

**Requirements:**

- Python 3.x
- Streamlit
- Sentence Transformers
- Faiss
- NumPy
- Pandas
- BeautifulSoup4 (for web scraping)
- Pygroq (for GROQ API integration) (optional)
- pymongo (for MongoDB connection) (optional)
- sqlite3 (for SQLite connection) (optional)
- Additional libraries for specific file types (e.g., docx, pptx)

**Installation:**

1. Clone this repository.
2. Create a virtual environment (recommended) and activate it.
3. Install the required libraries:

```bash
pip install streamlit pypdf sentence-transformers faiss-cpu numpy pandas requests beautifulsoup4 python-docx python-pptx python-dotenv pymongo sqlite3
```

**Usage:**

1. Set the `GROQ_API_KEY` environment variable (optional, required for GROQ integration).
2. Run the application:

```bash
streamlit run main.py
```

**Explanation of the Code:**

The code consists of several classes and functions:

- `DocumentProcessor`: This class handles document processing tasks, including text extraction for various file formats, data extraction from spreadsheets and JSON, splitting text into chunks, and building/saving the Faiss index.
-  `main` function: This function sets up the Streamlit application, checks for environment variables, manages document and URL processing, and handles database connections.

**Additional Notes:**

- This is a basic implementation and can be further customized for specific use cases.
- Error handling is included for various scenarios.
- The code utilizes caching and optimization techniques for faster processing.

**Feel free to contribute to this project by submitting pull requests or raising issues on Github.**
