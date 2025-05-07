from pypdf import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

def process_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    
    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=40, length_function=len, separator =" ")
    return splitter.split_text(text)

def get_relevant_chunks(question, db):
    # Effectue une recherche pour les chunks pertinents
    docs = db.similarity_search(question, k=3)
    relevant_chunks = [doc.page_content for doc in docs]
    
 
    return relevant_chunks



def get_llm_answer(question, context_chunks, api_key):
    # Si des chunks sont présents, assemble-les et génère une réponse
    context = "\n\n".join(context_chunks)
    
    # Construction du prompt à partir du contexte et de la question
    prompt = f"""
You are an intelligent assistant. Your goal is to answer the user's question by analyzing and summarizing the relevant information from the provided context. Do not copy the context word-for-word unless absolutely necessary. Always respond in the same language as the question.

Context:
{context}

Question:
{question}

If the answer is not clearly found in the context, respond with: "I did not find the answer in the document." or an equivalent phrase in the same language as the question.

Answer:
"""
    
    # Initialisation du modèle de chat
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo",
        temperature=0.1
    )
    
    # Préparation du message comme une liste d'objets HumanMessage
    messages = [HumanMessage(content=prompt)]
    
    # Appel du modèle avec la méthode `invoke` et récupération de la réponse
    response = llm.invoke(messages)
    
    # Retourner la réponse et le contexte
    return (response.content, context)

