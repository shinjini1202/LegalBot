import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rag_utils import retrieve_answer
import torch

# --- Page Configuration ---
st.set_page_config(page_title="⚖️ Domestic Violence Legal Q&A", layout="centered")

# --- Styling ---
st.markdown("""
<style>
    .stApp { background-color: #f9fafb; font-family: 'Segoe UI', sans-serif; }
    .answer-box { background-color: #eef3f8; padding: 1.5em; border-radius: 8px; border-left: 5px solid #d32f2f; color: #1a1a1a; }
    .context-box { background-color: #ffffff; padding: 1em; border-radius: 6px; margin-top: 0.5em; border: 1px solid #ddd; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Domestic Violence Legal Q&A Chatbot")

# --- Model Loading ---
@st.cache_resource
def load_model_assets():
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model

tokenizer, model = load_model_assets()

# --- File Upload ---
uploaded = st.file_uploader("📄 Upload a domestic violence case file (.txt)", type=["txt"])
if uploaded:
    new_case = uploaded.read().decode("utf-8", errors="ignore")
    st.session_state["case_text"] = new_case
    st.success("✅ File uploaded successfully!")

# --- Question Input ---
question = st.text_input("💬 Ask a legal question (e.g. 'How did the victim die?')")

# --- Execution Block ---
if st.button("Get Answer"):
    if "case_text" not in st.session_state:
        st.warning("⚠️ Please upload a case file first.")
    elif question.strip() == "":
        st.warning("⚠️ Please enter a question.")
    else:
        # 1. Retrieve the context from your knowledge base
        labels, relevance_score, context = retrieve_answer(question)
        
        with st.spinner("Analyzing documents and writing summary..."):
            # 2. Combine only the retrieved context (this is the relevant answer)
            full_context = "\n".join(context)

            # 3. Create a focused answer prompt using ONLY retrieved context
            prompt = f"""Answer the following legal question using ONLY the information in the provided legal context.Your answer must be direct and a fact-based summary.

===LEGAL CONTEXT===
{full_context}

===QUESTION===
{question}

===ANSWER==="""

            # 4. Generate Output
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=200, 
                min_length=20,
                num_beams=4, 
                repetition_penalty=2.0,
                length_penalty=1.2,
                early_stopping=True,
                temperature=0.7
            )
            
            ans = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # --- Display Results ---
            st.markdown("### 🧾 Answer")
            st.markdown(f"<div class='answer-box'>{ans}</div>", unsafe_allow_html=True)
            
            st.markdown("### 📊 Metadata")
            st.write(f"**Predicted Labels:** {', '.join(labels)}")
            st.write(f"**Relevance Score:** {relevance_score:.2f}")

            # --- Show supporting context snippets ---
            with st.expander("🔍 View Retrieved Source Contexts"):
                if len(context) > 0:
                    st.markdown("**Primary Context:**")
                    st.markdown(f"<div class='context-box'>{context[0]}</div>", unsafe_allow_html=True)
                if len(context) > 1:
                    st.markdown("**Secondary Context:**")
                    st.markdown(f"<div class='context-box'>{context[1]}</div>", unsafe_allow_html=True)