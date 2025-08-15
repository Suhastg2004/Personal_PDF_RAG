from pypdf import PdfReader
from typing import List, Dict
import re

def extract_pdf_text_with_metadata(pdf_path: str) -> List[Dict]:
    reader = PdfReader(pdf_path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r'\s+', ' ', text).strip()
        docs.append({
            "text": text,
            "metadata": {"source": pdf_path, "page": i+1}
        })
    return docs

def recursive_character_split(chunks: List[str], max_chars=800, overlap=100):
    # Accepts a single long text, returns list of overlapping chunks
    # You can extend with smarter split on \n, headings.
    out = []
    for text in chunks:
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            out.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
    return out

def chunk_documents(pages: List[Dict], max_chars=800, overlap=100) -> List[Dict]:
    results = []
    for page in pages:
        if not page["text"]:
            continue
        chunks = recursive_character_split([page["text"]], max_chars, overlap)
        for idx, ch in enumerate(chunks):
            md = dict(page["metadata"])
            md["chunk_id"] = idx
            results.append({"text": ch, "metadata": md})
    return results
