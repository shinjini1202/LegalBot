import streamlit as st
from transformers import pipeline
from rag_utils import retrieve_answer

st.set_page_config(page_title="⚖️ Domestic Violence Legal Q&A", layout="centered")

# --- Styling ---
st.markdown("""
<style>
    .stApp { background-color: #f9fafb; font-family: 'Segoe UI', sans-serif; }
    .answer-box { background-color: #eef3f8; padding: 1em; border-radius: 8px; }
    .context-box { background-color: #f4f4f6; padding: 0.7em; border-radius: 6px; margin-top: 0.5em; font-size: 0.9em; color: #333; }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Domestic Violence Legal Q&A Chatbot")

# --- Load Smaller, Faster Model ---
qa = pipeline("text2text-generation", model="google/flan-t5-base")

# --- File Upload ---
uploaded = st.file_uploader("📄 Upload a domestic violence case file (.txt)", type=["txt"])
if uploaded:
    new_case = uploaded.read().decode("utf-8", errors="ignore")
    st.session_state["case_text"] = new_case
    st.success("✅ File uploaded successfully!")

# --- Question Input ---
question = st.text_input("💬 Ask a legal question (e.g. 'Is there any demand for dowry?')")

# --- Answer Button ---
if st.button("Get Answer"):
    if "case_text" not in st.session_state:
        st.warning("⚠️ Please upload a case file first.")
    elif question.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        label, score, context = retrieve_answer(question)
        # Take more context (for better grounding)
        context_sample = " ".join(context[:4])
        case_snippet = st.session_state['case_text'][:1500]

        # --- Improved Prompt ---
        prompt = f"""
You are a legal research assistant specializing in domestic violence case analysis.

Use the CONTEXT section carefully to answer the QUESTION below.
If the context clearly supports the answer, rephrase and summarize it clearly in 2–3 lines.
If not enough evidence exists, respond: "Insufficient evidence in context."

### CONTEXT (from prior annotated data)
{context_sample}

### NEW CASE SNIPPET
{case_snippet}

### QUESTION
{question}

Answer in a structured, factual, and lawyer-friendly manner.
Keep it concise (2–3 lines) and avoid raw repetition.
"""

        ans = qa(prompt, max_length=250, temperature=0.3, num_return_sequences=1)[0]['generated_text']

        # --- Display ---
        st.markdown("### 🧾 Answer")
        st.markdown(f"<div class='answer-box'>{ans}</div>", unsafe_allow_html=True)
        st.caption(f"**Predicted Label:** {label} | **Relevance Score:** {score:.2f}")

        # --- Show supporting context snippets ---
        with st.expander("🔍 View Retrieved Source Contexts"):
            if len(context) > 0:
                st.markdown(f"<div class='context-primary'><b>Primary Context:</b> {context[0][:600]}...</div>", unsafe_allow_html=True)
            if len(context) > 1:
                st.markdown(f"<div class='context-support'><b>Supporting Context:</b> {context[1][:600]}...</div>", unsafe_allow_html=True)
