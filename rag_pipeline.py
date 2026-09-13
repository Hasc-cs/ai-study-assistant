"""
rag_pipeline.py
----------------
High-Capacity Backend RAG Pipeline for Large Documents (up to 100 MB).

Fixes applied:
1. Embedding Model: Uses 'models/gemini-embedding-001' (the active, supported model).
2. LLM Model: Uses 'gemini-1.5-flash' for high-quota reliability (1,500 requests/day).
3. Disk-backed streaming: Uses temporary files to prevent Streamlit Cloud OOM crashes.
4. Garbage collection & memory cleanup: Frees memory page-by-page.
5. In-Memory FAISS: Eliminates SQLite table locking errors.
"""

import gc
import io
import json
import os
import re
import tempfile
import time
from typing import List, Optional, Dict, Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --------------------------------------------------------------------------
# Configuration Constants
# --------------------------------------------------------------------------

# 'models/gemini-embedding-001' is the officially active model on v1beta
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-1.5-flash"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

EMBED_BATCH_SIZE = 30
EMBED_PAUSE_SECONDS = 0.5

MAX_RETRIES = 5
DEFAULT_RETRY_SECONDS = 20
MAX_PAGES_PER_DOC = 120


# --------------------------------------------------------------------------
# Rate-Limit & Backoff Helpers
# --------------------------------------------------------------------------

def _extract_retry_delay(error_text: str, default: int = DEFAULT_RETRY_SECONDS) -> int:
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", error_text)
    if match:
        return int(match.group(1)) + 2
    match = re.search(r"retry in ([\d.]+)s", error_text, re.IGNORECASE)
    if match:
        return int(float(match.group(1))) + 2
    return default


def _is_rate_limit_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return "429" in error_text or "quota" in lowered or "rate limit" in lowered


def _call_with_retry(func, *args, max_retries: int = MAX_RETRIES, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            error_text = str(exc)
            if _is_rate_limit_error(error_text) and attempt < max_retries:
                delay = _extract_retry_delay(error_text)
                time.sleep(delay)
                continue
            raise
    raise last_exc


# --------------------------------------------------------------------------
# Custom Exceptions
# --------------------------------------------------------------------------

class PDFExtractionError(Exception):
    pass


class VectorStoreError(Exception):
    pass


class GenerationError(Exception):
    pass


# --------------------------------------------------------------------------
# 1. Memory-Safe PDF Extraction
# --------------------------------------------------------------------------

def extract_documents_from_pdfs(
    uploaded_files: List[Any],
    max_pages: int = MAX_PAGES_PER_DOC,
) -> List[Document]:
    if not uploaded_files:
        raise PDFExtractionError("No files were provided for extraction.")

    documents: List[Document] = []
    failed_files: List[str] = []

    for uploaded_file in uploaded_files:
        filename = getattr(uploaded_file, "name", "document.pdf")
        temp_file_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file_path = temp_file.name
                if hasattr(uploaded_file, "read"):
                    uploaded_file.seek(0)
                    while chunk := uploaded_file.read(1024 * 1024):
                        temp_file.write(chunk)
                elif hasattr(uploaded_file, "getvalue"):
                    temp_file.write(uploaded_file.getvalue())

            reader = PdfReader(temp_file_path)
            total_pages = len(reader.pages)

            if total_pages == 0:
                failed_files.append(f"{filename} (empty PDF)")
                continue

            pages_to_read = min(total_pages, max_pages)
            pages_with_text = 0

            for page_index in range(pages_to_read):
                page_num = page_index + 1
                try:
                    page = reader.pages[page_index]
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

            del reader
            gc.collect()

            if pages_with_text == 0:
                failed_files.append(
                    f"{filename} (no readable text found — scanned or image-only PDF)"
                )

        except PdfReadError:
            failed_files.append(f"{filename} (corrupted PDF structure)")
        except Exception as exc:  # noqa: BLE001
            failed_files.append(f"{filename} ({exc})")
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass

    if not documents:
        details = "; ".join(failed_files) if failed_files else "unknown extraction error"
        raise PDFExtractionError(
            f"Could not extract readable text from uploaded file(s). Details: {details}"
        )

    extract_documents_from_pdfs.last_failed_files = failed_files  # type: ignore[attr-defined]
    return documents


# --------------------------------------------------------------------------
# 2. Text Chunking
# --------------------------------------------------------------------------

def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    if not documents:
        raise VectorStoreError("No documents available to chunk.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n# ",
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise VectorStoreError("Text chunking produced 0 segments.")

    return chunks


# --------------------------------------------------------------------------
# 3. Batched In-Memory FAISS Vector Indexing
# --------------------------------------------------------------------------

def build_vector_store(
    chunks: List[Document],
    api_key: str,
    persist_directory: Optional[str] = None,
):
    if not api_key:
        raise VectorStoreError("A Gemini API key is required to build embeddings.")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key,
        )

        first_batch = chunks[:EMBED_BATCH_SIZE]
        vector_store = _call_with_retry(
            FAISS.from_documents,
            documents=first_batch,
            embedding=embeddings,
        )

        if len(chunks) > EMBED_BATCH_SIZE:
            for i in range(EMBED_BATCH_SIZE, len(chunks), EMBED_BATCH_SIZE):
                next_batch = chunks[i : i + EMBED_BATCH_SIZE]
                _call_with_retry(vector_store.add_documents, documents=next_batch)
                time.sleep(EMBED_PAUSE_SECONDS)

        gc.collect()
        return vector_store

    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(
            f"Failed to build vector store. Details: {exc}"
        ) from exc


def get_llm(api_key: str, temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    if not api_key:
        raise GenerationError("A Gemini API key is required.")
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
# 4a. Grounded Question Answering with Citations
# --------------------------------------------------------------------------

def answer_question(
    query: str,
    vector_store,
    api_key: str,
    k: int = 4,
) -> Dict[str, Any]:
    if not query or not query.strip():
        raise GenerationError("Please enter a question.")
    if vector_store is None:
        raise VectorStoreError("No documents have been processed yet.")

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(query)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Retrieval failed: {exc}") from exc

    if not relevant_docs:
        return {
            "answer": "I could not find anything relevant to that question in the uploaded document.",
            "citations": [],
        }

    context_blocks = []
    for i, doc in enumerate(relevant_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_blocks.append(f"[Excerpt {i} | {source}, page {page}]\n{doc.page_content}")
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You are a university academic study assistant. Answer the student's question
using ONLY the textbook excerpts below.

Rules:
- If the answer is not contained in the context, explicitly state that the material
  does not cover it — do not invent facts.
- Be concise, clear, and exam-focused.
- Do not fabricate page numbers or citations.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

    llm = get_llm(api_key, temperature=0.2)
    try:
        response = _call_with_retry(llm.invoke, prompt)
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
# 4b. Quiz Generation
# --------------------------------------------------------------------------

def _gather_context(vector_store, topic: Optional[str], k: int = 8) -> str:
    try:
        search_query = topic.strip() if (topic and topic.strip()) else "key concepts principles definitions"
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(search_query)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to retrieve context: {exc}") from exc

    if not docs:
        raise VectorStoreError("No content available in the vector store.")

    blocks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{source}, page {page}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _extract_json_block(text: str) -> str:
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


def generate_quiz(
    vector_store,
    api_key: str,
    topic: Optional[str] = None,
    num_questions: int = 3,
) -> List[Dict[str, Any]]:
    if vector_store is None:
        raise VectorStoreError("No documents have been processed yet.")

    context_text = _gather_context(vector_store, topic, k=8)
    topic_instruction = (
        f"Focus specifically on the topic: '{topic.strip()}'."
        if topic and topic.strip()
        else "Cover a representative range of the textbook material below."
    )

    prompt = f"""You are an exam writer creating a practice quiz based strictly on the excerpts below.

{topic_instruction}

Create exactly {num_questions} multiple-choice questions. Each question must have
4 options (A, B, C, D), exactly one correct answer, and a one-sentence rationale.

Respond with ONLY a valid JSON array in exactly this format:
[
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "A",
    "explanation": "..."
  }}
]

TEXTBOOK MATERIAL:
{context_text}
"""

    llm = get_llm(api_key, temperature=0.3)
    try:
        response = _call_with_retry(llm.invoke, prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(f"Gemini failed to generate the quiz: {exc}") from exc

    json_text = _extract_json_block(raw_text)

    try:
        quiz_data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"Failed to parse quiz response: {exc}") from exc

    if not isinstance(quiz_data, list) or not quiz_data:
        raise GenerationError("No quiz questions were returned.")

    validated = []
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
            validated.append({
                "question": item["question"],
                "options": options,
                "correct_answer": item["correct_answer"],
                "explanation": item.get("explanation", ""),
            })

    if not validated:
        raise GenerationError("Generated quiz did not match required format.")

    return validated


# --------------------------------------------------------------------------
# 4c. Summarization
# --------------------------------------------------------------------------

def summarize_content(
    vector_store,
    api_key: str,
    topic: Optional[str] = None,
    num_points: int = 5,
) -> str:
    if vector_store is None:
        raise VectorStoreError("No documents have been processed yet.")

    context_text = _gather_context(vector_store, topic, k=10)
    topic_instruction = (
        f"Focus specifically on the topic: '{topic.strip()}'."
        if topic and topic.strip()
        else "Summarize the core textbook concepts."
    )

    prompt = f"""You are a revision coach summarizing textbook material before an exam.

{topic_instruction}

Using ONLY the textbook excerpts below, produce exactly {num_points} clear,
high-yield revision bullet points. Each bullet must:
- Start with a bolded core concept or term (**Concept Name**)
- Follow with a concise 1-2 sentence explanation
- Focus on high-probability exam definitions and rules

Respond with ONLY the Markdown bullet list, nothing else.

TEXTBOOK MATERIAL:
{context_text}
"""

    llm = get_llm(api_key, temperature=0.3)
    try:
        response = _call_with_retry(llm.invoke, prompt)
        summary_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:  # noqa: BLE001
        raise GenerationError(f"Failed to generate summary: {exc}") from exc

    return summary_text.strip()