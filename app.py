# app.py
# --- FIX PARA SQLITE EN STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import zipfile
from google import genai
from google.genai import types

# --- IMPORTS DE LANGCHAIN ---
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURACIÓN ---
MODELO = "gemini-2.5-flash" 
DB_PATH = "genio_db_knowledge"  
ZIP_PATH = "genio_db_knowledge.zip"
PDF_FOLDER = "pdfs"             
EMBEDDING_MODEL_NAME = "text-embedding-004"

# Asegurar que exista la carpeta de PDFs (necesaria para procesar cargas)
os.makedirs(PDF_FOLDER, exist_ok=True)

# --- DESCOMPRESIÓN AUTOMÁTICA ---
if not os.path.exists(DB_PATH) and os.path.exists(ZIP_PATH):
    print("📦 ZIP detectado. Descomprimiendo...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DB_PATH)

# --- PERSONALIDAD ---
SYSTEM_INSTRUCTION = """
Eres "Genio", un tutor socrático y asistente resolutivo.
1. MODO ENSEÑAR (Predeterminado): NUNCA dar la respuesta directa. Usa preguntas guía.
2. MODO RESOLVER (Guía Resolutivo): Dar la respuesta directa si se pide.
"""
chat_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.7,
    max_output_tokens=1000
)

# --- INICIALIZACIÓN ---
@st.cache_resource
def initialize_gemini():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("❌ Faltan secrets.")
        st.stop()
    client = genai.Client(api_key=api_key)
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME, google_api_key=api_key
    )
    return client, embedding_function

client, embedding_function = initialize_gemini()

def load_rag_database(): 
    global embedding_function 
    try:
        if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
            return Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
        return None
    except Exception:
        return None

if 'vector_db' not in st.session_state:
    st.session_state.vector_db = load_rag_database()

# --- FUNCIONES DE APRENDIZAJE EN VIVO ---
def process_new_file(uploaded_file):
    try:
        # Guardamos el archivo temporalmente para poder leerlo
        file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        if st.session_state.vector_db is None:
            st.session_state.vector_db = Chroma.from_documents(
                texts, embedding_function, persist_directory=DB_PATH
            )
        else:
            st.session_state.vector_db.add_documents(texts)
            
        return True, len(texts)
    except Exception as e:
        return False, str(e)

# --- FUNCIONES EXTRA ---
def get_rag_sources():
    if not st.session_state.vector_db: return []
    try:
        data = st.session_state.vector_db.get()
        metadatas = data.get('metadatas', [])
        unique = set([os.path.basename(m['source']) for m in metadatas if m and 'source' in m])
        return list(unique)
    except: return []

def generate_response(prompt):
    contexto_rag = ""
    if st.session_state.vector_db:
        try:
            docs = st.session_state.vector_db.similarity_search(prompt, k=3)
            contexto_rag = "\n\n".join([doc.page_content for doc in docs])
        except: pass
    
    prompt_ctx = f"[CONTEXTO RAG]: {contexto_rag}\n[PREGUNTA]: {prompt}"
    try:
        if 'chat_session' not in st.session_state:
            st.session_state.chat_session = client.chats.create(model=MODELO, config=chat_config)
        return st.session_state.chat_session.send_message(prompt_ctx).text
    except Exception as e: return f"Error: {e}"

# --- INTERFAZ ---
st.set_page_config(page_title="Genio Tutor", page_icon="🦉", layout="wide")

with st.sidebar:
    st.header("📂 Biblioteca")
    st.subheader("Subir nuevo conocimiento")
    uploaded_files = st.file_uploader("Añadir PDF a la sesión", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        for up_file in uploaded_files:
            if up_file.name not in [os.path.basename(s) for s in get_rag_sources()]:
                with st.spinner(f"Aprendiendo {up_file.name}..."):
                    success, info = process_new_file(up_file)
                    if success:
                        st.toast(f"✅ {up_file.name} aprendido", icon="🧠")
                    else:
                        st.error(f"Error: {info}")
    st.divider()
    if st.session_state.vector_db:
        st.success(f"✅ Memoria Activa")
        for s in get_rag_sources(): st.markdown(f"- 📄 `{s}`")
    else: st.error("❌ RAG Inactivo")

st.title("🦉 Genio: Tu Super Tutor")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Estoy listo para estudiar contigo. 📚"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner('Pensando...'):
        response_text = generate_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.messages.append(msg_data)

