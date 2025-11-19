# app.py
import os
import streamlit as st
from google import genai
from google.genai import types
# FIX 1: Importar Chroma de la ruta 'community' (soluciona ModuleNotFoundError)
from langchain_community.vectorstores import Chroma
# FIX 2: Importar la clase de Embeddings con el nombre completo (soluciona ImportError/AttributeError)
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

# --- CONFIGURACIÓN DE VARIABLES GLOBALES ---
MODELO = "gemini-2.5-flash" 
# Ruta donde Streamlit esperará encontrar la base de datos RAG (carpeta genio_db_knowledge)
DB_PATH = "genio_db_knowledge" 
EMBEDDING_MODEL_NAME = "text-embedding-004" 

# --- PERSONALIDAD DEL TUTOR (MODO DUAL) ---
SYSTEM_INSTRUCTION = """
Eres "Genio", un tutor socrático y asistente resolutivo. Tu objetivo es alternar entre dos modos, según lo solicite el usuario.

--- REGLAS DE MODO ---
1. MODO ENSEÑAR (Predeterminado):
   - Objetivo: Fomentar el aprendizaje y la autonomía.
   - Regla de Oro: NUNCA dar la respuesta directa. Usar preguntas guía y el método socrático. Ser motivador (emojis).

2. MODO RESOLVER (Guía Resolutivo):
   - Objetivo: Proveer la solución clara o un resumen de hechos.
   - Activación: Si el usuario dice 'Modo: Resolver', 'Dame la respuesta', o 'Resuélvelo'.
   - Regla de Oro: Ofrecer la respuesta directa, clara y paso a paso.

--- REGLAS GLOBALES ---
- Usa **negritas** para destacar las palabras clave.
- Utiliza el CONTEXTO RAG provisto para responder o guiar.
"""

# Configuración base del chat
chat_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.7,
    max_output_tokens=1000
)

# --- INICIALIZACIÓN DE RECURSOS (CACHÉ) ---

@st.cache_resource
def initialize_gemini():
    """Inicializa el cliente de Gemini y la función de embeddings."""
    # 1. Cargar API Key desde Streamlit Secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        # Se detiene si la clave no está en los secretos de Streamlit
        st.error("❌ ERROR: La clave GEMINI_API_KEY no se encontró en st.secrets.")
        st.stop()
        
    client = genai.Client(api_key=api_key)
    
    # 2. Inicializar Embeddings para RAG
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME, 
        google_api_key=api_key
    )
    return client, embedding_function

# Inicializar recursos globalmente (se ejecutan una vez)
client, embedding_function = initialize_gemini()


# FIX 3: La función de carga ya no recibe argumentos para evitar el UnhashableParamError
@st.cache_resource 
def load_rag_database(): 
    """Carga la base de datos vectorial ChromaDB desde el directorio."""
    
    # Accedemos a la variable global 'embedding_function' ya inicializada
    global embedding_function 
    
    try:
        # Verifica si el directorio existe y si tiene archivos
        if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
            db = Chroma(persist_directory=DB_PATH, 
                        embedding_function=embedding_function)
            st.success("✅ Base de datos RAG cargada con éxito.")
            return db
        else:
            st.warning("⚠️ Base de conocimiento RAG no encontrada. El tutor solo usará conocimiento general.")
            return None
    except Exception as e:
        st.error(f"❌ Error al cargar la BD: {e}. ¿Está la carpeta `{DB_PATH}` en el repositorio?")
        return None

# Carga la base de datos
vector_db = load_rag_database()

# Iniciar la sesión de chat de Gemini (Persiste entre interacciones)
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model=MODELO,
        config=chat_config
    )

# --- LÓGICA DE RESPUESTA (CON RAG) ---

def generate_response(prompt):
    """Genera la respuesta del tutor, aumentada con RAG."""
    
    # 1. Recuperación RAG
    contexto_rag = ""
    if vector_db:
        try:
            # Busca los 3 fragmentos más relevantes en la base de datos
            docs = vector_db.similarity_search(prompt, k=3)
            contexto_rag = "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            st.warning(f"Error en la consulta RAG: {e}")
    
    # 2. Inyección de Contexto y Pregunta
    # El SYSTEM_INSTRUCTION (Reglas de MODO) se aplica a este prompt
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

# --- INTERFAZ STREAMLIT ---

st.set_page_config(page_title="Genio Tutor RAG", page_icon="🦉", layout="wide")
st.title("🦉 Genio: Tu Super Tutor IA RAG")
st.markdown(f"Modelo: `{MODELO}` | Base de Datos: {'✅ Activa' if vector_db else '❌ Inactiva (Solo Conocimiento General)'}")

# Inicializar el historial de chat de Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_message = "¡Hola! Soy Genio. Estoy listo para ayudarte a aprender con el método socrático. Si quieres una respuesta directa, escribe 'Modo: Resolver'. 🧠"
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
