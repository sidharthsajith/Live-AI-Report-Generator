This is a Streamlit application that allows users to upload documents, crawl websites, and ask questions to retrieve relevant information.  The application utilizes sentence embeddings and a Faiss index for efficient retrieval.

## Features

- Upload documents in various formats (PDF, DOCX, TXT, PPTX, XLSX, CSV, JSON)
- Crawl websites to extract text content
- Integrate with MongoDB and SQLite databases (optional)
- Process documents and build a knowledge base using sentence embeddings
- Ask questions and receive concise, factual answers based on the uploaded documents, crawled data, and connected databases

## Requirements

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

## Installation

1. Clone this repository.
2. Create a virtual environment (recommended) and activate it.
3. Install the required libraries:

```bash
pip install streamlit pypdf sentence-transformers faiss-cpu numpy pandas requests beautifulsoup4 python-docx python-pptx python-dotenv pymongo sqlite3
```

##Usage

1. Set the `GROQ_API_KEY` environment variable (optional, required for GROQ integration).
2. Run the application:

```bash
streamlit run main.py
```

## Workflow

1. **Document/URL Input:**
   - Users can upload documents in various formats (PDF, DOCX, TXT, PPTX, XLSX, CSV, JSON) or provide URLs to websites.
2. **Text Extraction:**
   - The uploaded documents are processed to extract text content.
   - For websites, web scraping techniques are employed to extract text from HTML content.
3. **Text Chunking:**
   - The extracted text is divided into smaller chunks to improve processing efficiency and reduce memory usage.
4. **Embedding Generation:**
   - Each text chunk is converted into a numerical representation (embedding) using a sentence transformer model. This representation captures the semantic meaning of the text.
5. **Index Creation:**
   - The embeddings are added to a Faiss index, which is an efficient library for similarity search in high-dimensional spaces.
6. **Query Processing:**
   - When a user asks a question, the query is also converted into an embedding.
   - The Faiss index is queried to find the most similar text chunks to the query embedding.
7. **Contextual Understanding and Response Generation:**
   - The retrieved text chunks are used as context to generate a relevant and informative response.
   - A language model (e.g., Groq) is used to process the query and context, and generate a comprehensive answer.

## Code Breakdown

### DocumentProcessor Class

- **Initialization:**
  - Initializes the embedding model, Faiss index, and other necessary components.
  - Optionally connects to MongoDB and SQLite databases.
- **Text Extraction:**
  - Provides methods to extract text from various file formats (PDF, DOCX, TXT, PPTX, XLSX, CSV, JSON).
- **Text Chunking:**
  - Splits text into smaller chunks of a specified size.
- **Embedding Generation and Index Creation:**
  - Encodes text chunks into embeddings and adds them to the Faiss index.
- **Query Processing:**
  - Encodes the user query into an embedding.
  - Queries the Faiss index to retrieve the most similar text chunks.
  - Uses a language model (Groq) to generate a response based on the retrieved context.
- **Database Interactions:**
  - Provides methods to query MongoDB and SQLite databases to retrieve relevant information.
- **Web Crawling:**
  - Extracts text from websites using BeautifulSoup and requests libraries.

### main Function

- Sets up the Streamlit application.
- Handles user input for documents and URLs.
- Triggers document processing and indexing.
- Provides a text input field for user queries.
- Displays the generated response.
- Offers a report generation functionality.

**Key Points:**

- **Embedding Model:** The `SentenceTransformer` model is used to convert text into numerical representations.
- **Faiss Index:** This efficient similarity search index is used to quickly find relevant text chunks.
- **Language Model:** The Groq language model is used to generate human-quality responses based on the context.
- **Database Integration:** The application can optionally connect to MongoDB and SQLite databases to expand the knowledge base.
- **Web Crawling:** The application can crawl websites to extract relevant text content.
- **Report Generation:** The application can generate reports summarizing the query, answer, and sources.

**Overall, the code provides a robust and efficient framework for building a question-answering system powered by large language models and semantic search.**

**Additional Notes:**

- This is a basic implementation and can be further customized for specific use cases.
- Error handling is included for various scenarios.
- The code utilizes caching and optimization techniques for faster processing.

**Feel free to contribute to this project by submitting pull requests or raising issues on Github.**
