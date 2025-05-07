import streamlit as st
from utils import process_pdf, get_relevant_chunks, get_llm_answer
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Ask your PDF", layout="wide")
st.title("📄 Ask Questions about Your PDF")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
question = st.text_input("Ask a question")

if uploaded_file and question:
    with st.spinner("Processing..."):
        texts = process_pdf(uploaded_file)
        #st.write(f"{len(texts)} chunks générés")     #just for testing the splitter
        #st.write(texts[:5])  # Affiche les 5 premiers chunks

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.from_texts(texts, embeddings)

        relevant_chunks = get_relevant_chunks(question, db)
        answer, context = get_llm_answer(question, relevant_chunks, openai_api_key)


        #st.write (context)
        st.markdown("### Answer")
        st.write(answer)
