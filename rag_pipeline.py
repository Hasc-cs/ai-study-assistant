"""
rag_pipeline.py
----------------
Backend RAG (Retrieval-Augmented Generation) pipeline for the
AI Smart Study & Assignment Assistant.

Handles:
    - PDF text extraction with page-level metadata
    - Text chunking
    - Vector store creation/updating (ChromaDB + Gemini Embeddings)
    - Question answering with page citations
    - MCQ quiz generation
    - Topic summarization

All functions are designed to be called directly from a Streamlit
frontend (app.py) and raise informative exceptions that the UI layer
can catch and display with st.error / st.warning.
"""

import io
import json
import re
import uuid
from typing import List, Optional, Dict, Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-1.5-flash"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --------------------------------------------------------------------------
# Custom Exceptions
# --------------------------------------------------------------------------

class PDFExtractionError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""
    pass


class VectorStoreError(Exception):
    """Raised when the vector store cannot be built or queried."""
    pass


class GenerationError(Exception):
    """Raised when the LLM fails to produce a usable response."""
    pass


# --------------------------------------------------------------------------
# 1. PDF Extraction
# --------------------------------------------------------------------------

def extract_documents_from_pdfs(uploaded_files: List[Any]) -> List[Document]:
    """
    Extract text from a list of uploaded PDF files (Streamlit UploadedFile
    objects or file-like objects), preserving page numbers and source
    filenames in metadata.

    Args:
        uploaded_files: list of file-like objects with .name and .read()/.getvalue()

    Returns:
        List of langchain Document objects, one per non-empty page.

    Raises:
        PDFExtractionError: if a file cannot be parsed or no text at all
            could be extracted from any of the uploaded files.
    """
    if not uploaded_files:
        raise PDFExtractionError("No files were provided for extraction.")

    documents: List[Document] = []
    failed_files: List[str] = []

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "unknown.pdf")
        try:
            # Support both Streamlit UploadedFile (has getvalue) and raw bytes/file objects
            if hasattr(uploaded_file, "getvalue"):
                file_bytes = uploaded_file.getvalue()
            else:
                file_bytes = uploaded_file.read()

            reader = PdfReader(io.BytesIO(file_bytes))

            if len(reader.pages) == 0:
                failed_files.append(f"{filename} (no pages found)")
                continue

            pages_with_text = 0
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""

                page_text = page_text.strip()
                if page_text:
                    pages_with_text += 1
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={"page": page_num, "source": filename},
                        )
                    )

            if pages_with_text == 0:
                failed_files.append(
                    f"{filename} (no extractable text — likely a scanned/image-only PDF)"
                )

        except PdfReadError:
            failed_files.append(f"{filename} (corrupted or unreadable PDF)")
        except Exception as exc:  # noqa: BLE001
            failed_files.append(f"{filename} ({exc})")

    if not documents:
        details = "; ".join(failed_files) if failed_files else "unknown error"
        raise PDFExtractionError(
            f"Could not extract any readable text from the uploaded PDF(s). Details: {details}"
        )

    # Attach failed_files info onto the returned list via a side-channel attribute
    # so the caller (Streamlit UI) can warn about partial failures.
    extract_documents_from_pdfs.last_failed_files = failed_files  # type: ignore[attr-defined]

    return documents


# --------------------------------------------------------------------------
# 2. Chunking
# --------------------------------------------------------------------------

def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split page-level documents into smaller overlapping chunks while
    preserving the original page/source metadata on every chunk.
    """
    if not documents:
        raise VectorStoreError("No documents available to chunk.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise VectorStoreError("Text splitting produced zero chunks.")

    return chunks


# --------------------------------------------------------------------------
# 3. Vector Store
# --------------------------------------------------------------------------

def build_vector_store(
    chunks: List[Document],
    api_key: str,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Build a Chroma vector store from document chunks using Gemini embeddings.

    Args:
        chunks: list of chunked Document objects.
        api_key: Gemini API key.
        persist_directory: optional path for on-disk persistence. If None,
            an in-memory (ephemeral) Chroma collection is created.

    Returns:
        A Chroma vector store instance ready for similarity search.

    Raises:
        VectorStoreError: on embedding or index-build failure (e.g. bad API key).
    """
    if not api_key:
        raise VectorStoreError("A Gemini API key is required to build embeddings.")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )

        # Unique collection name avoids collisions across repeated runs in the
        # same Streamlit session / Chroma client.
        collection_name = f"study_assistant_{uuid.uuid4().hex[:8]}"

        kwargs: Dict[str, Any] = {
            "documents": chunks,
            "embedding": embeddings,
            "collection_name": collection_name,
        }
        if persist_directory:
            kwargs["persist_directory"] = persist_directory

        vector_store = Chroma.from_documents(**kwargs)
        return vector_store

    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(
            f"Failed to build the vector store. Check your Gemini API key and network "
            f"connection. Details: {exc}"
        ) from exc


def get_llm(api_key: str, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini chat model."""
    if not api_key:
        raise GenerationError("A Gemini API key is required to use the language model.")
    try:
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(f"Failed to initialize Gemini model: {exc}") from exc


# --------------------------------------------------------------------------
# 4a. Question Answering
# --------------------------------------------------------------------------

def answer_question(
    query: str,
    vector_store: Chroma,
    api_key: str,
    k: int = 4,
) -> Dict[str, Any]:
    """
    Answer a user's question using retrieval-augmented generation, strictly
    grounded in the uploaded documents, and return page-level citations.

    Returns:
        {
            "answer": str,
            "citations": [{"source": str, "page": int, "snippet": str}, ...]
        }
    """
    if not query or not query.strip():
        raise GenerationError("Please enter a non-empty question.")
    if vector_store is None:
        raise VectorStoreError("No documents have been processed yet.")

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(query)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Retrieval failed: {exc}") from exc

    if not relevant_docs:
        return {
            "answer": (
                "I couldn't find anything relevant to that question in the "
                "uploaded documents. Try rephrasing, or upload more material."
            ),
            "citations": [],
        }

    context_blocks = []
    for i, doc in enumerate(relevant_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_blocks.append(f"[Excerpt {i} | {source}, page {page}]\n{doc.page_content}")
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a meticulous study assistant. Answer the student's question
using ONLY the context excerpts below, which come from their own course materials.

Rules:
- If the answer is not contained in the context, say clearly that the materials
  don't cover it — do not make anything up.
- Be concise, clear, and exam-relevant.
- Do not fabricate page numbers or sources; only use what is given in the context.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

    llm = get_llm(api_key, temperature=0.2)
    try:
        response = llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(f"Gemini failed to generate an answer: {exc}") from exc

    citations = []
    seen = set()
    for doc in relevant_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        snippet = doc.page_content[:280].strip()
        if len(doc.page_content) > 280:
            snippet += "..."
        citations.append({"source": source, "page": page, "snippet": snippet})

    return {"answer": answer_text.strip(), "citations": citations}


# --------------------------------------------------------------------------
# Shared helper: gather context for a topic (or the whole corpus)
# --------------------------------------------------------------------------

def _gather_context(
    vector_store: Chroma,
    topic: Optional[str],
    k: int = 8,
) -> str:
    """Retrieve representative context chunks for a topic, or a broad sample
    of the corpus if no topic is given."""
    try:
        if topic and topic.strip():
            retriever = vector_store.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(topic)
        else:
            # No topic: pull a broad sample of the whole collection.
            collection = vector_store.get()
            documents_raw = collection.get("documents", []) or []
            metadatas_raw = collection.get("metadatas", []) or []
            docs = []
            for text, meta in list(zip(documents_raw, metadatas_raw))[:k]:
                docs.append(Document(page_content=text, metadata=meta or {}))
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to retrieve context: {exc}") from exc

    if not docs:
        raise VectorStoreError("No content available in the vector store to work with.")

    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{source}, page {page}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _extract_json_block(text: str) -> str:
    """Pull the first JSON array/object out of an LLM response that may be
    wrapped in markdown code fences or extra prose."""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)

    bracket_match = re.search(r"(\[.*\])", text, re.DOTALL)
    if bracket_match:
        return bracket_match.group(1)

    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        return brace_match.group(1)

    return text


# --------------------------------------------------------------------------
# 4b. Quiz Generation
# --------------------------------------------------------------------------

def generate_quiz(
    vector_store: Chroma,
    api_key: str,
    topic: Optional[str] = None,
    num_questions: int = 3,
) -> List[Dict[str, Any]]:
    """
    Generate multiple-choice quiz questions grounded in the uploaded documents.

    Returns:
        A list of dicts, each shaped as:
        {
            "question": str,
            "options": {"A": str, "B": str, "C": str, "D": str},
            "correct_answer": "A" | "B" | "C" | "D",
            "explanation": str
        }
    """
    if vector_store is None:
        raise VectorStoreError("No documents have been processed yet.")

    context_text = _gather_context(vector_store, topic, k=8)

    topic_instruction = (
        f"Focus specifically on the topic: '{topic.strip()}'."
        if topic and topic.strip()
        else "Cover a broad, representative range of the material below."
    )

    prompt = f"""You are an expert exam-writer creating a practice quiz for a university
student, based strictly on the course material excerpts below.

{topic_instruction}

Create exactly {num_questions} multiple-choice questions. Each question must have
4 options (A, B, C, D), exactly one correct answer, and a one-sentence explanation
of why that answer is correct, grounded in the material.

Respond with ONLY a valid JSON array (no markdown, no prose, no code fences) in
exactly this shape:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A",
    "explanation": "..."
  }}
]

COURSE MATERIAL:
{context_text}
"""

    llm = get_llm(api_key, temperature=0.4)
    try:
        response = llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(f"Gemini failed to generate the quiz: {exc}") from exc

    json_text = _extract_json_block(raw_text)

    try:
        quiz_data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise GenerationError(
            f"The model returned an unparsable quiz format. Please try again. ({exc})"
        ) from exc

    if not isinstance(quiz_data, list) or not quiz_data:
        raise GenerationError("The model did not return any quiz questions.")

    # Validate structure defensively.
    validated: List[Dict[str, Any]] = []
    for item in quiz_data:
        if not isinstance(item, dict):
            continue
        options = item.get("options", {})
        if (
            "question" in item
            and isinstance(options, dict)
            and set(["A", "B", "C", "D"]).issubset(options.keys())
            and item.get("correct_answer") in ["A", "B", "C", "D"]
        ):
            validated.append(
                {
                    "question": item["question"],
                    "options": options,
                    "correct_answer": item["correct_answer"],
                    "explanation": item.get("explanation", ""),
                }
            )

    if not validated:
        raise GenerationError("The generated quiz did not match the expected format.")

    return validated


# --------------------------------------------------------------------------
# 4c. Summarization
# --------------------------------------------------------------------------

def summarize_content(
    vector_store: Chroma,
    api_key: str,
    topic: Optional[str] = None,
    num_points: int = 5,
) -> str:
    """
    Produce a structured, exam-focused summary of the uploaded material
    (optionally scoped to a topic) as clean Markdown bullet points.

    Returns:
        A Markdown-formatted string with `num_points` bullet points.
    """
    if vector_store is None:
        raise VectorStoreError("No documents have been processed yet.")

    context_text = _gather_context(vector_store, topic, k=10)

    topic_instruction = (
        f"Focus specifically on the topic: '{topic.strip()}'."
        if topic and topic.strip()
        else "Summarize the key ideas across the entire uploaded material."
    )

    prompt = f"""You are a study coach helping a student revise efficiently before an exam.

{topic_instruction}

Using ONLY the course material excerpts below, produce exactly {num_points} clear,
high-yield bullet points a student could use for last-minute revision. Each bullet
should:
- Start with a short bolded key term or concept (Markdown **bold**)
- Be followed by a concise, plain-English explanation (1-2 sentences)
- Focus on concepts most likely to appear on an exam

Respond with ONLY the Markdown bullet list, nothing else.

COURSE MATERIAL:
{context_text}
"""

    llm = get_llm(api_key, temperature=0.3)
    try:
        response = llm.invoke(prompt)
        summary_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(f"Gemini failed to generate the summary: {exc}") from exc

    return summary_text.strip()
