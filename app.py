# app.py
import os
import streamlit as st
from google import genai
from google.genai import types
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

# --- CONFIGURACIÓN DE VARIABLES GLOBALES ---
MODELO = "gemini-2.5-flash" 
DB_PATH = "genio_db_knowledge" 
EMBEDDING_MODEL_NAME = "text-embedding-004" 

# --- PERSONALIDAD DEL TUTOR ---
SYSTEM_INSTRUCTION = """
Eres "Genio", un tutor socrático y asistente resolutivo. Tu objetivo es alternar entre dos modos, según lo solicite el usuario.

--- REGLAS DE MODO ---
1. MODO ENSEÑAR (Predeterminado):
   - Objetivo: Fomentar el aprendizaje y la autonomía.
   - Regla de Oro: NUNCA dar la respuesta directa. Usar preguntas guía y el método socrático.

2. MODO RESOLVER (Guía Resolutivo):
   - Objetivo: Proveer la solución clara o un resumen de hechos.
   - Activación: Si el usuario dice 'Modo: Resolver', 'Dame la respuesta', o 'Resuélvelo'.
   - Regla de Oro: Ofrecer la respuesta directa, clara y paso a paso.

--- REGLAS GLOBALES ---
- Usa **negritas** para destacar las palabras clave.
- Utiliza el CONTEXTO RAG provisto para responder o guiar.
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
        st.error("❌ ERROR: La clave GEMINI_API_KEY no se encontró en st.secrets.")
        st.stop()
        
    client = genai.Client(api_key=api_key)
    
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME, 
        google_api_key=api_key
    )
    return client, embedding_function

client, embedding_function = initialize_gemini()

@st.cache_resource 
def load_rag_database(): 
    global embedding_function 
    try:
        if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
            db = Chroma(persist_directory=DB_PATH, 
                        embedding_function=embedding_function)
            return db
        else:
            return None
    except Exception as e:
        return None

vector_db = load_rag_database()

# --- NUEVA FUNCIÓN: Obtener lista de archivos ---
def get_rag_sources():
    if not vector_db:
        return []
    try:
        # Obtenemos los metadatos de todos los documentos
        data = vector_db.get()
        metadatas = data.get('metadatas', [])
        
        # Extraemos los nombres de archivo únicos ('source')
        unique_sources = set()
        for m in metadatas:
            if m and 'source' in m:
                # Solo guardamos el nombre del archivo, no la ruta completa
                unique_sources.add(os.path.basename(m['source']))
        
        return list(unique_sources)
    except Exception as e:
        return []

# Inicializar Chat
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = client.chats.create(
        model=MODELO,
        config=chat_config
    )

# --- LÓGICA DE RESPUESTA ---
def generate_response(prompt):
    contexto_rag = ""
    if vector_db:
        try:
            docs = vector_db.similarity_search(prompt, k=3)
            contexto_rag = "\n\n".join([doc.page_content for doc in docs])
        except Exception:
            pass
    
    prompt_con_contexto = f"""
    [CONTEXTO RAG]: {contexto_rag}
    [PREGUNTA]: {prompt}
    """
    
    try:
        response = st.session_state.chat_session.send_message(prompt_con_contexto)
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- INTERFAZ STREAMLIT ---

st.set_page_config(page_title="Genio Tutor", page_icon="🦉", layout="wide")

# === BARRA LATERAL (SIDEBAR) CON ARCHIVOS ===
with st.sidebar:
    st.header("📂 Base de Conocimiento")
    st.markdown("---")
    if vector_db:
        st.success("✅ Base de Datos Activa")
        sources = get_rag_sources()
        if sources:
            st.markdown(f"**📚 {len(sources)} Documentos Indexados:**")
            for source in sources:
                st.markdown(f"- 📄 `{source}`")
        else:
            st.info("No se detectaron nombres de archivos en la metadata.")
    else:
        st.error("❌ RAG Inactivo")
        st.markdown("El tutor está usando solo su conocimiento general.")
        
    st.markdown("---")
    st.caption("v1.2 - Streamlit Cloud")

# === ÁREA PRINCIPAL ===
st.title("🦉 Genio: Tu Super Tutor IA")
st.markdown("¡Hola! Soy Genio. Pregúntame sobre tus documentos y aprenderemos juntos. 🧠")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hola, ¿qué quieres aprender hoy?"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tu pregunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('Consultando base de conocimientos...'):
        response_text = generate_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
