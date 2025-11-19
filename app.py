# app.py
import os
import streamlit as st
from google import genai
from google.genai import types
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv # Util para pruebas locales, aunque Streamlit usa st.secrets

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---

# ATENCIÓN: Streamlit Cloud requiere que la clave se guarde en st.secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    # Si falla la búsqueda de secrets (ej. en pruebas locales), muestra un error
    st.error("❌ ERROR: La clave GEMINI_API_KEY no se encontró en st.secrets.")
    st.stop()


# Variables globales del tutor
MODELO = "gemini-2.5-flash" 
# ¡IMPORTANTE! Se asume que esta carpeta (con los archivos RAG) está en el mismo repo de GitHub.
DB_PATH = "genio_db_knowledge" 
EMBEDDING_MODEL_NAME = "text-embedding-004" 

# --- PERSONALIDAD DEL TUTOR (Modo Dual) ---
SYSTEM_INSTRUCTION = """
Eres "Genio", un tutor socrático y asistente resolutivo. Tu objetivo es alternar entre dos modos, según lo solicite el usuario.

--- REGLAS DE MODO ---
1. MODO ENSEÑAR (Predeterminado): NUNCA dar la respuesta directa. Usar preguntas guía y el método socrático.
2. MODO RESOLVER (Guía Resolutivo): Si el usuario dice 'Modo: Resolver' o 'Resuélvelo', da la respuesta directa y paso a paso.

--- REGLAS GLOBALES ---
- Usa **negritas** para destacar las palabras clave.
- Utiliza el CONTEXTO RAG provisto para responder o guiar.
"""

# Configuración del chat
chat_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.7,
    max_output_tokens=1000
)

# --- Inicialización de Recursos (Cacheados para Rendimiento) ---

@st.cache_resource
def initialize_gemini():
    """Inicializa el cliente de Gemini y la función de embeddings."""
    client = genai.Client(api_key=API_KEY)
    
    # Inicialización de Embeddings para RAG
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME, 
        google_api_key=API_KEY
    )
    return client, embedding_function

@st.cache_resource
def load_rag_database(embedding_function):
    """Carga la base de datos vectorial ChromaDB."""
    try:
        if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
            db = Chroma(persist_directory=DB_PATH, 
                        embedding_function=embedding_function)
            st.success("✅ Base de datos RAG cargada con éxito.")
            return db
        else:
            st.warning("⚠️ Base de conocimiento RAG no encontrada. El tutor solo usará conocimiento general.")
            return None
    except Exception as e:
        st.error(f"❌ Error al cargar la BD: {e}. ¿Está el folder `{DB_PATH}` en el repositorio?")
        return None

# Inicializar recursos
client, embedding_function = initialize_gemini()
vector_db = load_rag_database(embedding_function)

# Iniciar la sesión de chat de Gemini (Solo la primera vez)
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model=MODELO,
        config=chat_config
    )

# --- 2. LÓGICA DE RESPUESTA (CON RAG) ---

def generate_response(prompt):
    """Genera la respuesta del tutor, aumentada con RAG."""
    
    # 1. Recuperación RAG
    contexto_rag = ""
    if vector_db:
        try:
            docs = vector_db.similarity_search(prompt, k=3)
            contexto_rag = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            st.warning(f"Error en la consulta RAG: {e}")
    
    # 2. Inyección de Contexto y Pregunta
    prompt_con_contexto = f"""
    [CONTEXTO DE TU BASE DE DATOS RAG]: {contexto_rag}
    [PREGUNTA DEL ALUMNO]: {prompt}
    """
    
    # 3. Respuesta de Gemini
    try:
        response = st.session_state.chat_session.send_message(prompt_con_contexto)
        return response.text
    except Exception as e:
        return f"⚠️ Error ({MODELO}): No se pudo generar la respuesta. Detalle: {str(e)}"

# --- 3. INTERFAZ STREAMLIT ---

st.set_page_config(page_title="Genio Tutor RAG", page_icon="🦉")
st.title("🦉 Genio: Tu Super Tutor IA RAG")
st.markdown(f"Modelo: `{MODELO}` | Base de Datos: {'✅ Activa' if vector_db else '❌ Inactiva (Solo Conocimiento General)'}")

# Inicializar el historial de chat de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_message = "¡Hola! Soy Genio. Estoy listo para ayudarte a aprender con el método socrático. ¿Qué te gustaría estudiar? 🧠"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

# Mostrar historial de mensajes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de chat
if prompt := st.chat_input("Escribe aquí tu pregunta o tarea..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.spinner('Genio está pensando...'):
        response_text = generate_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response_text)
    

    st.session_state.messages.append({"role": "assistant", "content": response_text})
