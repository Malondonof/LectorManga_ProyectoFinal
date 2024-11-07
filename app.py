import os
import streamlit as st
import pdfplumber  # Usamos pdfplumber para leer el PDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import openai

# Estilo en la aplicación
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

# Entrada para la clave de API de OpenAI
api_key = st.text_input('Ingresa tu Clave de API de OpenAI', type='password')
os.environ['OPENAI_API_KEY'] = api_key

# Cargar archivo PDF
uploaded_pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

if uploaded_pdf:
    # Leer el PDF con pdfplumber
    with pdfplumber.open(uploaded_pdf) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # Extraemos el texto de cada página

    # Si el texto extraído está vacío, mostramos un mensaje de error
    if not text:
        st.error("No se pudo extraer texto del archivo PDF. Intenta con otro PDF.")

    else:
        # Dividir el texto en fragmentos (chunks)
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
        chunks = text_splitter.split_text(text)

        # Crear embeddings a partir de los fragmentos del texto
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Mostrar el campo de entrada para las preguntas
        st.subheader("Escribe lo que quieres saber sobre el documento")
        user_question = st.text_area(" ")

        if user_question:
            # Realizar búsqueda en la base de conocimientos
            docs = knowledge_base.similarity_search(user_question)

            # Cargar el modelo de lenguaje y realizar la cadena de preguntas y respuestas
            llm = OpenAI(model_name="gpt-4")
            chain = load_qa_chain(llm, chain_type="stuff")

            # Mostrar la respuesta
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                st.write(response)
