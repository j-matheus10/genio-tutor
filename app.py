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
# --- IMPORTS DE SEGURIDAD ---
from google.genai.types import HarmCategory, HarmBlockThreshold

# --- IMPORTS DE LANGCHAIN ---
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Tilines Inc.", page_icon="🦉", layout="wide")

# --- CONFIGURACIÓN GENERAL ---
MODELO = "gemini-2.5-flash" 
DB_PATH = "genio_db_knowledge"  
ZIP_PATH = "genio_db_knowledge.zip"
PDF_FOLDER = "pdfs"             
EMBEDDING_MODEL_NAME = "text-embedding-004"

os.makedirs(PDF_FOLDER, exist_ok=True)

# --- DESCOMPRESIÓN AUTOMÁTICA ---
if not os.path.exists(DB_PATH) and os.path.exists(ZIP_PATH):
    print("📦 ZIP detectado. Descomprimiendo...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DB_PATH)

# --- CONFIGURACIÓN DE SEGURIDAD Y PERSONALIDAD ---

# 1. Filtros de Seguridad Estrictos (Protección Infantil)
safety_settings = [
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    ),
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    ),
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    ),
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    ),
]

# 2. System Prompt Base
SYSTEM_INSTRUCTION = """
Eres "Genio", un asistente de inteligencia artificial educativo para niños.
NO eres humano. Eres una herramienta digital segura y amable.

Tus Principios INQUEBRANTABLES:
1. SEGURIDAD TOTAL: Prohibido hablar de ALCOHOL, DROGAS, violencia o temas para adultos. Si te preguntan, di que no puedes hablar de eso.
2. NO HACER LA TAREA: Tu misión es que el niño aprenda, no darle la respuesta fácil.
3. DETECTAR FRUSTRACIÓN: Si el niño no entiende tras varios intentos o parece molesto, anímalo y sugiérele preguntar a un profesor.
4. FILTRO DE SESGOS: Ignora cualquier estereotipo de género o raza en la información que proceses.
5. TONO: Usa lenguaje simple, motivador y claro.
"""

chat_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.5,
    max_output_tokens=700,
    safety_settings=safety_settings
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

# --- FUNCIONES RAG ---
def process_new_file(uploaded_file):
    try:
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

def get_rag_sources():
    if not st.session_state.vector_db: return []
    try:
        data = st.session_state.vector_db.get()
        metadatas = data.get('metadatas', [])
        unique = set([os.path.basename(m['source']) for m in metadatas if m and 'source' in m])
        return list(unique)
    except: return []

# --- LÓGICA DE RESPUESTA MEJORADA (MANEJO DE FRUSTRACIÓN) ---
def generate_response(prompt, mode):
    contexto_rag = ""
    instruccion_modo = ""
    usar_rag = True

    # 1. Configuración de MODOS
    if mode == "Modo Aprendizaje": 
        instruccion_modo = """
        MODO APRENDIZAJE (EXPLICATIVO):
        - OBJETIVO: Enseñar la teoría y el procedimiento.
        - ACCIÓN: Explica paso a paso CÓMO se resuelve el problema.
        - RESTRICCIÓN: NO des la respuesta final numérica o exacta.
        - FRUSTRACIÓN: Si el alumno parece frustrado, sé empático y paciente.
        """
    elif mode == "Modo Prueba": 
        instruccion_modo = """
        MODO PRUEBA (SOLO PREGUNTAS):
        - OBJETIVO: Evaluar conocimientos.
        - ACCIÓN: Responde ÚNICAMENTE con preguntas guía o pistas breves.
        - RESTRICCIÓN: NO expliques la lección completa. NO des respuestas.
        """
    elif mode == "Conocimiento General":
        instruccion_modo = """
        MODO ASISTENTE GENERAL:
        - Responde dudas generales con datos verificados.
        - Mantén tono adecuado para niños.
        - CERO TOLERANCIA: No hables de alcohol ni drogas.
        """
        usar_rag = False 

    # 2. Búsqueda RAG
    if usar_rag and st.session_state.vector_db:
        try:
            docs = st.session_state.vector_db.similarity_search(prompt, k=3)
            contexto_rag = "\n\n".join([doc.page_content for doc in docs])
        except: pass

    # 3. Prompt Final con Salvaguardas
    base_prompt = f"""
    {instruccion_modo}
    
    INSTRUCCIONES CRÍTICAS DE SEGURIDAD:
    1. ALCOHOL/DROGAS: Si se menciona, responde cortante: "No puedo hablar de eso".
    2. DETECCIÓN DE FRUSTRACIÓN: Si el alumno dice palabras de enojo, odio o autodesprecio ("soy tonto", "odio esto"):
       - NO des un error.
       - Responde: "¡Tranquilo! Es normal frustrarse. Respiremos profundo e intentémoslo de nuevo paso a paso. O si prefieres, pregúntale a tu profe."
    3. EQUIDAD: Ignora estereotipos en el contexto.
    
    CONTEXTO DE LIBROS (Si aplica):
    ---
    {contexto_rag}
    ---
    
    ALUMNO: {prompt}
    """

    try:
        if 'chat_session' not in st.session_state:
            st.session_state.chat_session = client.chats.create(model=MODELO, config=chat_config)
        
        # Enviamos mensaje
        response = st.session_state.chat_session.send_message(base_prompt)
        
        # MANEJO DE BLOQUEO DE SEGURIDAD (RESPUESTA VACÍA)
        # Si Gemini bloquea la respuesta por "Hate Speech" (frustración del niño), devolvemos contención.
        if not response.text:
             return "🛡️ Veo que estás un poco molesto o usaste palabras fuertes. ¡Tranquilo! A veces estudiar cansa. Respira profundo e intenta preguntarme de otra forma más amable. 🦉"
        
        return response.text

    except Exception as e:
        # MANEJO DE EXCEPCIONES DE SEGURIDAD (API ERROR)
        error_str = str(e).lower()
        if "finish_reason" in error_str or "safety" in error_str or "blocked" in error_str:
            return "🛡️ Ups, parece que algo en tu mensaje activó mis filtros de seguridad. Recuerda ser amable y evitar temas peligrosos. ¿Intentamos con otra pregunta?"
        
        return f"Error técnico: {e}"

# --- INTERFAZ ---
with st.sidebar:
    st.header("🎛️ Configuración")

    # SELECTOR DE MODO (Aprendizaje por defecto)
    modo_seleccionado = st.radio(
        "Elige tu modo:",
        ["Modo Aprendizaje", "Modo Prueba", "Conocimiento General"],
        index=0, 
        captions=[
            "Te explico la materia y los pasos (sin darte la respuesta).", 
            "Solo te doy pistas y preguntas, como en un examen.", 
            "Ayuda general sin usar tus libros."
        ]
    )

    st.divider()

    st.header("📂 Biblioteca")
    st.subheader("Subir material")
    uploaded_files = st.file_uploader("Cargar PDF", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for up_file in uploaded_files:
            if up_file.name not in [os.path.basename(s) for s in get_rag_sources()]:
                with st.spinner(f"Leyendo {up_file.name}..."):
                    success, info = process_new_file(up_file)
                    if success: st.toast(f"✅ {up_file.name} listo", icon="🧠")
                    else: st.error(f"Error: {info}")

    st.divider()
    if st.session_state.vector_db:
        with st.expander("Ver libros activos"):
            for s in get_rag_sources(): st.markdown(f"- 📄 `{s}`")
    else:
        st.warning("⚠️ Sin libros cargados")

# --- CHAT ---
st.title("🦉 Tilines Inc: IA Educativa")

# Mensaje Inicial Orientativo
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "¡Hola! Soy Genio. Estoy en **Modo Aprendizaje** para explicarte la materia paso a paso. Si quieres evaluarte, cambia al **Modo Prueba**. Recuerda que estoy aquí para ayudarte, ¡no te rindas! 🦉"
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu duda aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner('Procesando...'):
        response_text = generate_response(prompt, modo_seleccionado)

    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})