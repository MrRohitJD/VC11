"""Email document chunking using LangChain RecursiveCharacterTextSplitter.

Implements configurable splitting policies:
- Short emails (< chunk_size) preserved as single atomic chunks
- Long emails split into overlapping chunks preserving thread and sender context
- Strict metadata inheritance on every chunk
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.preprocessing import clean_email_text

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


class EmailChunker:
    """Configurable email chunking pipeline using LangChain splitters."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            keep_separator=True,
        )

    def split_document(self, document: Document) -> List[Document]:
        """Split a LangChain email document into chunks with preserved metadata."""
        subject = document.metadata.get("subject", "")
        sender = document.metadata.get("sender", "")
        email_id = document.metadata.get("email_id", "")
        raw_text = document.page_content

        # Clean the text
        cleaned_text = clean_email_text(raw_text)
        if not cleaned_text:
            return []

        # If text is short enough, keep as single chunk
        if len(cleaned_text) <= self.chunk_size:
            chunk_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
            meta = dict(document.metadata)
            meta["chunk_index"] = 0
            meta["chunk_strategy"] = "atomic_single"
            meta["content_hash"] = chunk_hash
            header_prefix = f"Subject: {subject}\nFrom: {sender}\n\n" if subject or sender else ""
            return [Document(page_content=f"{header_prefix}{cleaned_text}", metadata=meta)]

        # Long email: split into manageable chunks
        splits = self.splitter.split_text(cleaned_text)
        chunks: List[Document] = []
        for idx, split_text in enumerate(splits):
            chunk_hash = hashlib.sha256(split_text.encode("utf-8")).hexdigest()
            meta = dict(document.metadata)
            meta["chunk_index"] = idx
            meta["chunk_strategy"] = "recursive_character"
            meta["content_hash"] = chunk_hash
            header_prefix = f"Subject: {subject} [Part {idx + 1}]\nFrom: {sender}\n\n" if subject or sender else ""
            chunks.append(Document(page_content=f"{header_prefix}{split_text}", metadata=meta))

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split a batch of LangChain documents into chunks."""
        all_chunks: List[Document] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks

    def chunk_email(
        self,
        text: str,
        email_id: Any,
        user_id: Any = None,
        subject: str = "",
        sender: str = "",
    ) -> List[Any]:
        """Chunk raw email text returning chunk objects with index and hash."""
        doc = Document(
            page_content=text,
            metadata={
                "email_id": str(email_id),
                "user_id": str(user_id) if user_id else "",
                "subject": subject,
                "sender": sender,
            },
        )
        split_docs = self.split_document(doc)

        class ChunkResult:
            def __init__(self, d: Document):
                self.chunk_index = d.metadata.get("chunk_index", 0)
                self.chunk_text = d.page_content
                self.content_hash = d.metadata.get("content_hash", "")
                self.metadata = d.metadata

        return [ChunkResult(d) for d in split_docs]


default_chunker = EmailChunker()
