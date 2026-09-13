import io
import math
import os
import re
import uuid
import asyncio
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".markdown"}

load_dotenv(APP_DIR.parent / ".env")

app = FastAPI(title="Atlas RAG")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'-]{1,}", text.lower())


def chunk_text(text: str, size: int = 850, overlap: int = 140) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return chunks


def extract_text(filename: str, raw: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return raw.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(raw)).pages)
    if suffix == ".docx":
        from docx import Document
        try:
            return "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs)
        except Exception:
            # Some Word-compatible exporters omit the package relationship that
            # python-docx expects. The document body is still usually available
            # as WordprocessingML inside the .docx ZIP container.
            return extract_docx_body(raw)
    raise ValueError("Unsupported file type")


def extract_docx_body(raw: bytes) -> str:
    """Extract paragraph text from a DOCX body without relying on relationships."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
    except (KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise ValueError("The file is not a readable Word document") from exc

    paragraphs = []
    # Strict OOXML uses purl.oclc.org namespaces while older DOCX files use
    # schemas.openxmlformats.org. Match element local names to support both.
    local_name = lambda tag: tag.rsplit("}", 1)[-1]
    for paragraph in (node for node in root.iter() if local_name(node.tag) == "p"):
        text = "".join(node.text or "" for node in paragraph.iter() if local_name(node.tag) == "t")
        if text.strip():
            paragraphs.append(text)
    return "\n".join(paragraphs)


class Retriever:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.chunks: list[dict[str, Any]] = []
        self.df: Counter[str] = Counter()

    def _rebuild_stats(self) -> None:
        self.df = Counter()
        for chunk in self.chunks:
            self.df.update(set(chunk["terms"]))

    def add(self, filename: str, text: str) -> dict[str, Any]:
        doc_id = str(uuid.uuid4())[:8]
        pieces = chunk_text(text)
        self.docs[doc_id] = {"id": doc_id, "name": filename, "chunks": len(pieces), "characters": len(text)}
        for idx, piece in enumerate(pieces):
            self.chunks.append({"id": f"{doc_id}-{idx}", "doc_id": doc_id, "text": piece, "terms": tokens(piece)})
        self._rebuild_stats()
        return self.docs[doc_id]

    def remove(self, doc_id: str) -> None:
        if doc_id not in self.docs:
            raise KeyError(doc_id)
        del self.docs[doc_id]
        self.chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
        self._rebuild_stats()

    def search(self, question: str, limit: int = 5) -> list[dict[str, Any]]:
        query = tokens(question)
        if not query or not self.chunks:
            return []
        n = len(self.chunks)
        q_counts = Counter(query)
        q_norm = math.sqrt(sum(v * v for v in q_counts.values()))
        scored = []
        for chunk in self.chunks:
            counts = Counter(chunk["terms"])
            dot = sum(q_counts[t] * (counts[t] * (math.log((n + 1) / (self.df[t] + 1)) + 1)) for t in q_counts)
            norm = math.sqrt(sum((v * (math.log((n + 1) / (self.df[t] + 1)) + 1)) ** 2 for t, v in counts.items()))
            score = dot / (q_norm * norm) if norm else 0
            if score:
                scored.append({**chunk, "score": score, "source": self.docs[chunk["doc_id"]]["name"]})
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]


store = Retriever()


class ChatRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = []


def fallback_answer(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "I couldn't find a relevant passage in the documents you've uploaded. Try rephrasing the question or adding a source that covers it."
    excerpts = []
    for hit in hits[:3]:
        sentence = re.split(r"(?<=[.!?])\s+", hit["text"])[0]
        excerpts.append(f"From **{hit['source']}**: {sentence}")
    return "Here’s what I found in your documents:\n\n" + "\n\n".join(excerpts) + "\n\nSet `OPENAI_API_KEY` to enable a synthesized answer grounded in these sources."


async def generate_answer(question: str, hits: list[dict[str, Any]]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback_answer(hits)
    from google import genai
    from google.genai import types
    context = "\n\n".join(f"[{i + 1}] {h['source']}: {h['text']}" for i, h in enumerate(hits))
    prompt = f"""Answer the question using only the supplied document excerpts. Be concise and precise. Cite claims with [1], [2], etc. If the excerpts do not answer it, say so clearly.\n\nQuestion: {question}\n\nExcerpts:\n{context}"""
    client = genai.Client(api_key=api_key)
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text or "I couldn't generate an answer."


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/documents")
async def list_documents() -> list[dict[str, Any]]:
    return list(store.docs.values())


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or Path(file.filename).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, "Upload a PDF, DOCX, TXT, or Markdown file.")
    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(413, "Files must be 15 MB or smaller.")
    try:
        text = extract_text(file.filename, raw)
    except Exception as exc:
        raise HTTPException(422, f"Could not read this document: {exc}") from exc
    if len(text.strip()) < 30:
        raise HTTPException(422, "This document has too little extractable text.")
    return store.add(file.filename, text)


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str) -> dict[str, bool]:
    try:
        store.remove(doc_id)
    except KeyError:
        raise HTTPException(404, "Document not found")
    return {"ok": True}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, Any]:
    if not request.question.strip():
        raise HTTPException(400, "Ask a question first.")
    hits = store.search(request.question)
    answer = await generate_answer(request.question, hits)
    sources = [{"id": hit["id"], "name": hit["source"], "excerpt": hit["text"], "score": round(hit["score"], 2)} for hit in hits]
    return {"answer": answer, "sources": sources, "mode": "ai" if os.getenv("GEMINI_API_KEY") else "retrieval"}
