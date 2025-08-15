# PDF RAG Chatbot

Chat with your PDFs using a local hybrid retriever (BM25 + embeddings) and Together AI for generation.

## Requirements
- Linux/macOS, Python 3.10
- Virtualenv (recommended)

## Setup
1. Create/activate a venv (example uses the existing `myenv`):
	```bash
	source ~/Downloads/Algorizz_assignment/myenv/bin/activate
	```
2. Install dependencies:
	```bash
	pip install --upgrade pip setuptools wheel
	pip install -r pdf-rag-chatbot/requirements.txt
	# If PyTorch install fails on Linux, try CPU wheel:
	# pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
	```
3. Add a `.env` file under `pdf-rag-chatbot/`:
	```
	TOGETHER_API_KEY=your_together_key
	# Optional, to force a model that works well on Free tier:
	TOGETHER_MODEL=mistralai/Mistral-7B-Instruct-v0.2
	# Optional for Hugging Face downloads:
	# HUGGINGFACE_HUB_TOKEN=your_hf_token
	```

## Run
```bash
source ~/Downloads/Algorizz_assignment/myenv/bin/activate
streamlit run pdf-rag-chatbot/app.py --server.port 8501
```
Open http://localhost:8501

Steps in the UI
1. Upload 1–2 PDFs in the sidebar.
2. Click "Build Index" (or just click "Ask"; it will auto-build).
3. Ask a question.

Notes
- Retrieval is local; no DB keys needed. To persist Chroma across runs, change to `chromadb.PersistentClient(path=".chroma")`.
- If you see a 403 from Together, set `TOGETHER_MODEL` to a smaller model (e.g., `google/gemma-2b-it` or `mistralai/Mistral-7B-Instruct-v0.2`) or add credits.
- First run may download models; this can take a few minutes.

## Deploy to Hugging Face Spaces
- Add a repo secret `TOGETHER_API_KEY`.
- Allow internet in Space settings for Together API calls.
- Ensure `requirements.txt` is at project root or set the working dir accordingly.
