import os
import streamlit as st
from PIL import Image
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import platform

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400&family=Lexend:wght@600&display=swap');

    h1, h2, h3 {
        font-family: 'Lexend', sans-serif;
    }

    p, div, label, span, input, textarea {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Título y subtítulo de la aplicación
st.title('LectorManga')

# Barra lateral con información
with st.sidebar:
   st.subheader("Aquí podrás escuchar descripciones detalladas del manga que estás leyendo")
