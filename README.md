# 📘 HR Policy Assistant — RAG + Streamlit

An AI-powered HR Policy Assistant that lets users upload an HR policy PDF and ask questions about it.

The application uses **Retrieval-Augmented Generation (RAG)**:

```text
HR Policy PDF
     ↓
PyMuPDF
     ↓
Text extraction
     ↓
Chunking
     ↓
Sentence Transformers
     ↓
Embeddings
     ↓
FAISS vector search
     ↓
Top relevant policy chunks
     ↓
Groq API
     ↓
OpenAI GPT-OSS 20B
     ↓
Answer + retrieved sources
```

## Features

- Upload an HR policy PDF directly in the Streamlit interface
- Extract PDF text using PyMuPDF
- Split the policy into overlapping chunks
- Create semantic embeddings using Sentence Transformers
- Store/search embeddings with FAISS
- Retrieve the most relevant policy sections
- Generate grounded answers using Groq's `openai/gpt-oss-20b`
- Display retrieved page/chunk sources
- No database required
- No local server required for deployment
- Deploy directly from GitHub to Streamlit Community Cloud

## Project files

```text
hr-policy-assistant/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Technology stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- NumPy
- Groq API
- OpenAI GPT-OSS 20B

## Important security note

Do **not** put your Groq API key inside `app.py`.

For Streamlit Community Cloud, store the key in the app's **Secrets** settings.

Use:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Never commit a real API key to GitHub.

## How RAG works

### 1. PDF upload

The user uploads an HR policy PDF through Streamlit.

### 2. Text extraction

PyMuPDF reads each PDF page and extracts selectable text.

### 3. Chunking

The text is split into overlapping chunks so that relevant policy information can be retrieved without sending the entire document to the LLM.

### 4. Embeddings

Sentence Transformers converts each chunk into a numerical vector.

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

### 5. FAISS

FAISS stores the vectors and performs similarity search.

The application normalizes vectors and uses inner-product search, which is equivalent to cosine similarity for normalized embeddings.

### 6. Retrieval

When the user asks a question, the question is also embedded.

FAISS retrieves the five most relevant policy chunks.

### 7. Generation

The retrieved chunks are sent to:

```text
openai/gpt-oss-20b
```

through the Groq API.

The model is instructed to answer only from the retrieved policy context.

## Deployment without VS Code, Colab, or Terminal

You can create the GitHub repository and upload the files entirely through your browser.

### Step 1 — Create a GitHub repository

1. Open GitHub.
2. Sign in.
3. Click **+** in the top-right corner.
4. Click **New repository**.
5. Repository name:

```text
hr-policy-assistant
```

6. Choose **Public** if you want the simplest Streamlit Community Cloud setup.
7. Click **Create repository**.

### Step 2 — Upload the project files

Inside the new repository:

1. Click **Add file**.
2. Click **Upload files**.
3. Upload:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
4. Click **Commit changes**.

Your repository should look like:

```text
hr-policy-assistant
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Step 3 — Create a Groq API key

Create a Groq API key from the Groq developer console.

Keep the key private.

Do not paste it into `app.py`.

### Step 4 — Open Streamlit Community Cloud

Open:

https://share.streamlit.io/

Sign in with GitHub and authorize Streamlit to access your repository.

### Step 5 — Deploy

1. Click **Create app**.
2. Select **Yup, I have an app**.
3. Select your GitHub repository:

```text
hr-policy-assistant
```

4. Branch:

```text
main
```

5. Main file path:

```text
app.py
```

6. Open **Advanced settings**.
7. Select Python **3.12** if available.
8. In **Secrets**, paste:

```toml
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

9. Click **Save**.
10. Click **Deploy**.

Streamlit will install the dependencies from `requirements.txt` and build the application.

### Step 6 — Test the application

After deployment:

1. Open your Streamlit app URL.
2. Upload an HR policy PDF.
3. Click **Process PDF**.
4. Ask a question such as:

```text
How many annual leave days are allowed?
```

5. Check the answer.
6. Open **Retrieved policy sources** to see the policy chunks used for the answer.

## Important limitations

### Scanned PDFs

The current version extracts selectable PDF text.

If the uploaded PDF is only scanned images, PyMuPDF may return little or no text. OCR should be added in a future version for scanned documents.

### Privacy

HR policies can contain sensitive company information. Do not upload confidential documents to an app unless your organization's privacy/security requirements allow it.

The application sends retrieved policy text to the Groq API for answer generation.

### Temporary document storage

This version keeps the processed document/index in the Streamlit session. It does not create a permanent vector database.

If the app restarts, users need to upload/process the PDF again.

## Updating the application

You do not need Git or a terminal.

Simply:

1. Open the GitHub repository.
2. Open `app.py`.
3. Click the pencil/edit button.
4. Make your changes.
5. Click **Commit changes**.

Streamlit Community Cloud automatically detects repository changes and redeploys the application.

## Future improvements

Possible production upgrades:

- OCR for scanned PDFs
- Multiple PDF support
- Persistent vector database
- Authentication/login
- Admin dashboard
- Conversation history
- Better source citations
- Department-specific policies
- Policy version management
- Feedback buttons
- Hybrid keyword + vector retrieval
- Reranking
- Multi-document RAG
