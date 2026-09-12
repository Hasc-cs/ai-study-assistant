"""
app.py
------
AI Smart Study & Assignment Assistant
A Streamlit RAG application that lets students upload course PDFs and:
    1. Ask questions with page-level citations
    2. Generate practice MCQ quizzes
    3. Get quick topic summaries

Run locally with:
    streamlit run app.py
"""

import os
import traceback

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    extract_documents_from_pdfs,
    chunk_documents,
    build_vector_store,
    answer_question,
    generate_quiz,
    summarize_content,
    PDFExtractionError,
    VectorStoreError,
    GenerationError,
)

load_dotenv()

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Smart Study & Assignment Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Custom CSS — glassmorphism / 3D floating effect
# --------------------------------------------------------------------------

st.markdown("""
<style>
/* Glassmorphism 3D floating effect for cards/containers */
div.stChatMessage, div.stForm {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    transition: transform 0.3s ease;
}

div.stChatMessage:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 35px rgba(79, 70, 229, 0.3);
}

/* 3D button styling with active click depth */
.stButton>button {
    background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
    color: white;
    border-radius: 10px;
    border: none;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
    transition: all 0.2s ease-in-out;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
}

.stButton>button:active {
    transform: translateY(1px);
    box-shadow: 0 2px 10px rgba(79, 70, 229, 0.4);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------

def init_session_state():
    defaults = {
        "vector_store": None,
        "documents_processed": False,
        "processed_filenames": [],
        "chat_history": [],       # list of {"role": ..., "content": ..., "citations": [...]}
        "quiz_data": None,        # list of question dicts
        "quiz_submitted": False,
        "quiz_selections": {},    # {question_index: selected_option}
        "summary_text": None,
        "gemini_api_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 Study Assistant")
    st.caption("Upload your course materials to get started.")

    st.markdown("### 🔑 Gemini API Key")
    env_key = os.getenv("GEMINI_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Enter your Gemini API key",
        value="",
        type="password",
        placeholder="Uses .env GEMINI_API_KEY if left blank",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )
    effective_api_key = api_key_input.strip() or env_key
    st.session_state["gemini_api_key"] = effective_api_key

    if effective_api_key:
        st.success("API key loaded ✅", icon="✅")
    else:
        st.warning("No API key found. Enter one above or set GEMINI_API_KEY in a .env file.")

    st.divider()

    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF slides or textbooks",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF files at once.",
    )

    process_clicked = st.button(
        "⚙️ Process Documents", use_container_width=True, type="primary"
    )

    if process_clicked:
        if not effective_api_key:
            st.error("Please provide a Gemini API key before processing documents.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF file first.")
        else:
            with st.spinner("Reading PDFs, chunking text, and building your knowledge base..."):
                try:
                    raw_documents = extract_documents_from_pdfs(uploaded_files)

                    failed_files = getattr(
                        extract_documents_from_pdfs, "last_failed_files", []
                    )

                    chunks = chunk_documents(raw_documents)
                    vector_store = build_vector_store(chunks, api_key=effective_api_key)

                    st.session_state["vector_store"] = vector_store
                    st.session_state["documents_processed"] = True
                    st.session_state["processed_filenames"] = list(
                        {f.name for f in uploaded_files}
                    )
                    # Reset downstream state on a fresh processing run
                    st.session_state["chat_history"] = []
                    st.session_state["quiz_data"] = None
                    st.session_state["quiz_submitted"] = False
                    st.session_state["quiz_selections"] = {}
                    st.session_state["summary_text"] = None

                    st.success(
                        f"✅ Processed {len(uploaded_files)} file(s) into "
                        f"{len(chunks)} searchable chunks!"
                    )
                    if failed_files:
                        st.warning(
                            "Some pages/files had issues:\n- " + "\n- ".join(failed_files)
                        )

                except PDFExtractionError as e:
                    st.error(f"📄 PDF extraction failed: {e}")
                except VectorStoreError as e:
                    st.error(f"🧠 Vector store error: {e}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ Unexpected error while processing documents: {e}")
                    with st.expander("Show technical details"):
                        st.code(traceback.format_exc())

    st.divider()

    if st.session_state["documents_processed"]:
        st.markdown("### ✅ Knowledge Base Ready")
        for name in st.session_state["processed_filenames"]:
            st.markdown(f"- {name}")
        if st.button("🗑️ Clear Documents", use_container_width=True):
            st.session_state["vector_store"] = None
            st.session_state["documents_processed"] = False
            st.session_state["processed_filenames"] = []
            st.session_state["chat_history"] = []
            st.session_state["quiz_data"] = None
            st.session_state["quiz_submitted"] = False
            st.session_state["quiz_selections"] = {}
            st.session_state["summary_text"] = None
            st.rerun()
    else:
        st.info("No documents processed yet. Upload PDFs and click **Process Documents**.")

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------

st.title("🎓 AI Smart Study & Assignment Assistant")
st.caption(
    "Turn your lecture slides and textbooks into an interactive study companion — "
    "ask questions, generate quizzes, and get instant summaries, all grounded in "
    "*your own* course material."
)

tab_ask, tab_quiz, tab_summary = st.tabs(
    ["💬 Ask Your Notes", "📝 Practice Quiz", "📌 Key Summaries"]
)

ready = st.session_state["documents_processed"] and st.session_state["vector_store"] is not None

# ==========================================================================
# TAB 1 — Ask Your Notes
# ==========================================================================
with tab_ask:
    st.subheader("Ask questions about your uploaded material")

    if not ready:
        st.info("👈 Upload and process your PDFs in the sidebar to start chatting.")
    else:
        # Render chat history
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("citations"):
                    with st.expander("📎 View source citations"):
                        for c in msg["citations"]:
                            st.markdown(
                                f"**{c['source']} — page {c['page']}**\n\n"
                                f"> {c['snippet']}"
                            )

        user_query = st.chat_input("Ask a question about your notes...")

        if user_query:
            st.session_state["chat_history"].append(
                {"role": "user", "content": user_query}
            )
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Searching your notes..."):
                    try:
                        result = answer_question(
                            query=user_query,
                            vector_store=st.session_state["vector_store"],
                            api_key=st.session_state["gemini_api_key"],
                        )
                        st.markdown(result["answer"])
                        if result["citations"]:
                            with st.expander("📎 View source citations"):
                                for c in result["citations"]:
                                    st.markdown(
                                        f"**{c['source']} — page {c['page']}**\n\n"
                                        f"> {c['snippet']}"
                                    )
                        st.session_state["chat_history"].append(
                            {
                                "role": "assistant",
                                "content": result["answer"],
                                "citations": result["citations"],
                            }
                        )
                    except (VectorStoreError, GenerationError) as e:
                        error_msg = f"⚠️ {e}"
                        st.error(error_msg)
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "content": error_msg, "citations": []}
                        )
                    except Exception as e:  # noqa: BLE001
                        st.error(f"❌ Unexpected error: {e}")
                        with st.expander("Show technical details"):
                            st.code(traceback.format_exc())

        if st.session_state["chat_history"]:
            if st.button("🧹 Clear chat history"):
                st.session_state["chat_history"] = []
                st.rerun()

# ==========================================================================
# TAB 2 — Practice Quiz
# ==========================================================================
with tab_quiz:
    st.subheader("Generate a practice quiz from your material")

    if not ready:
        st.info("👈 Upload and process your PDFs in the sidebar to generate a quiz.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            quiz_topic = st.text_input(
                "Optional: focus the quiz on a specific topic",
                placeholder="e.g. Photosynthesis, Chapter 3, Linked Lists...",
                key="quiz_topic_input",
            )
        with col2:
            st.write("")
            st.write("")
            generate_clicked = st.button(
                "🎲 Generate Quiz", use_container_width=True, type="primary"
            )

        if generate_clicked:
            with st.spinner("Writing your practice questions..."):
                try:
                    quiz = generate_quiz(
                        vector_store=st.session_state["vector_store"],
                        api_key=st.session_state["gemini_api_key"],
                        topic=quiz_topic,
                        num_questions=3,
                    )
                    st.session_state["quiz_data"] = quiz
                    st.session_state["quiz_submitted"] = False
                    st.session_state["quiz_selections"] = {}
                    st.success("Quiz ready! Scroll down to answer the questions.")
                except (VectorStoreError, GenerationError) as e:
                    st.error(f"⚠️ {e}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ Unexpected error generating quiz: {e}")
                    with st.expander("Show technical details"):
                        st.code(traceback.format_exc())

        if st.session_state["quiz_data"]:
            st.divider()
            quiz = st.session_state["quiz_data"]

            with st.form("quiz_form"):
                for idx, q in enumerate(quiz):
                    st.markdown(f"**Q{idx + 1}. {q['question']}**")
                    option_labels = [f"{key}. {val}" for key, val in q["options"].items()]
                    option_keys = list(q["options"].keys())

                    selected = st.radio(
                        f"Select an answer for question {idx + 1}",
                        options=option_keys,
                        format_func=lambda k, opts=q["options"]: f"{k}. {opts[k]}",
                        key=f"quiz_q_{idx}",
                        label_visibility="collapsed",
                        index=None,
                    )
                    st.session_state["quiz_selections"][idx] = selected
                    st.markdown("---")

                submitted = st.form_submit_button(
                    "✅ Submit Quiz", type="primary", use_container_width=True
                )
                if submitted:
                    st.session_state["quiz_submitted"] = True

            if st.session_state["quiz_submitted"]:
                score = 0
                total = len(quiz)
                for idx, q in enumerate(quiz):
                    selected = st.session_state["quiz_selections"].get(idx)
                    correct = q["correct_answer"]
                    is_correct = selected == correct
                    if is_correct:
                        score += 1

                    if selected is None:
                        st.warning(f"Q{idx + 1}: No answer selected.")
                    elif is_correct:
                        st.success(f"Q{idx + 1}: Correct! ({selected})")
                    else:
                        st.error(f"Q{idx + 1}: Incorrect. You chose {selected}, correct answer is {correct}.")

                    with st.expander(f"Explanation for Q{idx + 1}"):
                        st.markdown(q.get("explanation", "No explanation provided."))

                st.divider()
                percentage = round((score / total) * 100) if total else 0
                st.metric("Your Score", f"{score} / {total}", f"{percentage}%")

                if percentage == 100:
                    st.balloons()

# ==========================================================================
# TAB 3 — Key Summaries
# ==========================================================================
with tab_summary:
    st.subheader("Get a quick revision summary")

    if not ready:
        st.info("👈 Upload and process your PDFs in the sidebar to generate summaries.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            summary_topic = st.text_input(
                "Optional: focus the summary on a specific topic",
                placeholder="Leave blank to summarize everything",
                key="summary_topic_input",
            )
        with col2:
            st.write("")
            st.write("")
            summarize_clicked = st.button(
                "📌 Summarize", use_container_width=True, type="primary"
            )

        if summarize_clicked:
            with st.spinner("Condensing your material into key points..."):
                try:
                    summary = summarize_content(
                        vector_store=st.session_state["vector_store"],
                        api_key=st.session_state["gemini_api_key"],
                        topic=summary_topic,
                        num_points=5,
                    )
                    st.session_state["summary_text"] = summary
                except (VectorStoreError, GenerationError) as e:
                    st.error(f"⚠️ {e}")
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ Unexpected error generating summary: {e}")
                    with st.expander("Show technical details"):
                        st.code(traceback.format_exc())

        if st.session_state["summary_text"]:
            st.divider()
            st.markdown(st.session_state["summary_text"])
            st.download_button(
                "⬇️ Download Summary (.md)",
                data=st.session_state["summary_text"],
                file_name="study_summary.md",
                mime="text/markdown",
            )

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with Streamlit, LangChain, Google Gemini & ChromaDB · "
    "AI Smart Study & Assignment Assistant"
)
