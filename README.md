# 🎓 AI Smart Study & Assignment Assistant

A Retrieval-Augmented Generation (RAG) web app that turns any student's PDF
slides or textbooks into an interactive study companion — grounded, cited,
and instantly quizzable.

Built for a 48-hour AI Productivity Hackathon.

---

## 1. Problem Statement

University students routinely juggle hundreds of pages of lecture slides and
textbook chapters before exams. The typical workflow is painfully manual:

- Re-reading entire decks to find one answer to a specific question
- Manually writing practice questions with no guarantee they're representative
- Spending hours condensing chapters into revision notes
- Losing track of *where* in the material a fact actually came from, making
  it hard to verify answers or cite sources in assignments

**AI Smart Study & Assignment Assistant** solves this by letting a student
upload their own course PDFs and instantly:

1. **Ask questions** and get answers strictly grounded in their material,
   with exact page-level citations (no hallucinated facts).
2. **Generate practice MCQ quizzes** on demand, scoped to any topic.
3. **Get 5-bullet revision summaries** of a chapter or the whole course pack.

Everything is retrieved from the student's *own* uploaded content — not
generic web knowledge — so answers are always relevant to their specific
course and professor.

---

## 2. Architecture / Flow

```
                    ┌─────────────────────────┐
                    │   Streamlit Frontend     │
                    │        (app.py)          │
                    │  Sidebar: API key + PDFs │
                    │  Tabs: Ask / Quiz / Sum. │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │      rag_pipeline.py           │
                 │                                 │
                 │  1. PDF Upload                  │
                 │     └─ pypdf → per-page text     │
                 │        + {source, page} metadata │
                 │                                 │
                 │  2. Chunking                    │
                 │     └─ RecursiveCharacterText-   │
                 │        Splitter (1000 / 150)     │
                 │                                 │
                 │  3. Embedding                   │
                 │     └─ Gemini text-embedding-004 │
                 │                                 │
                 │  4. Vector Store                │
                 │     └─ ChromaDB (in-memory)      │
                 │                                 │
                 │  5. Retrieval + Generation       │
                 │     └─ Chroma similarity search  │
                 │        → Gemini 1.5 Flash        │
                 │        → answer_question()       │
                 │        → generate_quiz()         │
                 │        → summarize_content()     │
                 └───────────────┬─────────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │   Google Gemini API             │
                 │   (Embeddings + gemini-1.5-flash)│
                 └───────────────────────────────┘
```

**Flow in words:**
Student uploads PDFs → text is extracted page-by-page with metadata →
split into overlapping ~1000-character chunks → embedded via Gemini and
stored in a ChromaDB vector index → user questions/requests trigger a
similarity search against that index → the top matching chunks (with their
page/source metadata) are passed to Gemini 1.5 Flash as grounded context →
Gemini returns an answer, quiz, or summary, and the app renders citations
back to the exact page they came from.

---

## 3. Project Structure

```
study-assistant/
├── app.py              # Streamlit UI: sidebar, 3 tabs, session state
├── rag_pipeline.py      # Backend: PDF parsing, chunking, vector store, LLM calls
├── requirements.txt     # Pinned, compatible dependencies
├── .env                 # (you create this) GEMINI_API_KEY=...
└── README.md
```

---

## 4. Local Setup Instructions

### Step 1 — Clone / download the project
```bash
git clone <your-repo-url>
cd study-assistant
```

### Step 2 — Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Get a free Gemini API key
Visit [Google AI Studio](https://aistudio.google.com/app/apikey) and generate
a free API key.

### Step 5 — Configure your API key
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```
*(Alternatively, paste the key directly into the sidebar text field at
runtime — no `.env` file needed for local demos.)*

### Step 6 — Run the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

### Step 7 — Use it
1. Paste your API key in the sidebar (or rely on `.env`).
2. Upload one or more PDF files.
3. Click **Process Documents**.
4. Switch between the **Ask Your Notes**, **Practice Quiz**, and
   **Key Summaries** tabs.

---

## 5. Deploying to Streamlit Community Cloud

1. Push this project to a public (or private) GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your
   GitHub account.
3. Select the repo, branch, and `app.py` as the entry point.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```
5. Deploy. Streamlit Cloud will install `requirements.txt` automatically.

No GPU, external database, or paid infrastructure is required — Chroma runs
in-memory within the app process, and Gemini Flash keeps inference costs and
latency low.

---

## 6. Quantified Impact (for Hackathon Judges)

| Metric | Manual Approach | With AI Study Assistant | Est. Improvement |
|---|---|---|---|
| Time to find an answer in a 100-page deck | ~5–10 min of skimming | ~10–15 sec (chat query) | **~95% faster** |
| Time to build a 3-question practice quiz | ~15–20 min per topic | ~10–15 sec | **~98% faster** |
| Time to summarize a chapter for revision | ~20–30 min | ~15 sec | **~97% faster** |
| Citation accuracy / traceability | Manual, error-prone | Every claim traceable to exact page | Near-zero source ambiguity |
| Est. weekly time saved per student (3 courses) | — | **3–5 hours/week** | Reinvested into deeper study or rest |

**Why this matters:** for a student carrying 4–5 courses, even a conservative
3 hours/week saved translates to **12+ hours per month** — nearly a full
extra study day — without sacrificing depth, since every answer remains
grounded in the actual assigned material rather than generic AI knowledge.

---

## 7. Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Orchestration | LangChain |
| LLM | Google Gemini 1.5 Flash |
| Embeddings | Gemini `text-embedding-004` |
| Vector Store | ChromaDB |
| PDF Parsing | pypdf |
| Config | python-dotenv |

---

## 8. Error Handling & Guardrails

- Missing/invalid API key → clear `st.error` before any processing begins.
- Empty upload / no file selected → `st.warning`, no crash.
- Corrupted, empty, or scanned/image-only PDFs → skipped individually with a
  warning listing exactly which file/page failed, while still processing the
  remaining valid files.
- LLM returns malformed JSON (quiz generation) → caught and surfaced as a
  friendly retry prompt rather than a raw traceback.
- All unexpected exceptions are caught at the UI layer with an expandable
  "technical details" section for debugging, without breaking the app.

---

## 9. Future Enhancements

- OCR fallback (e.g. `pytesseract`) for scanned/image-only PDFs.
- Persistent, per-user Chroma storage across sessions.
- Difficulty-level tuning for quizzes (easy / medium / hard).
- Export quizzes and summaries as printable PDFs.
- Multi-language support for non-English course material.
