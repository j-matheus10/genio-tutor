# app.py
# --- FIX PARA SQLITE EN STREAMLIT CLOUD ---
# Esto evita errores de versión de base de datos en la nube
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- IMPORTS ---
import os
import streamlit as st
import fitz  # PyMuPDF para renderizar PDFs como imágenes
import io
from PIL import Image
from google import genai
from google.genai import types

# Importaciones corregidas para las nuevas versiones de librerías
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

# --- CONFIGURACIÓN DE VARIABLES GLOBALES ---
MODELO = "gemini-2.5-flash" 
DB_PATH = "genio_db_knowledge"  # Carpeta de la base de datos
PDF_FOLDER = "pdfs"             # Carpeta con los archivos PDF originales
EMBEDDING_MODEL_NAME = "text-embedding-004"

# --- PERSONALIDAD DEL TUTOR (MODO DUAL) ---
SYSTEM_INSTRUCTION = """
Eres "Genio", un tutor socrático y asistente resolutivo. Tu objetivo es alternar entre dos modos, según lo solicite el usuario.

--- REGLAS DE MODO ---
1. MODO ENSEÑAR (Predeterminado):
   - Objetivo: Fomentar el aprendizaje y la autonomía.
   - Regla de Oro: NUNCA dar la respuesta directa. Usar preguntas guía y el método socrático.
   - Tono: Amable, paciente y motivador.

2. MODO RESOLVER (Guía Resolutivo):
   - Objetivo: Proveer la solución clara o un resumen de hechos.
   - Activación: Si el usuario dice 'Modo: Resolver', 'Dame la respuesta', o 'Resuélvelo'.
   - Regla de Oro: Ofrecer la respuesta directa, clara y paso a paso.

--- REGLAS GLOBALES ---
- Usa **negritas** para destacar las palabras clave.
- Utiliza el CONTEXTO RAG provisto para responder o guiar.
"""

# Configuración del chat de Gemini
chat_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.7,
    max_output_tokens=1000
)

# --- INICIALIZACIÓN DE RECURSOS (CACHÉ) ---

@st.cache_resource
def initialize_gemini():
    """Inicializa el cliente de Gemini y la función de embeddings."""
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

# Inicializamos recursos globales para evitar problemas de caché
client, embedding_function = initialize_gemini()

@st.cache_resource 
def load_rag_database(): 
    """Carga la base de datos vectorial ChromaDB."""
    # Usamos la variable global para evitar 'UnhashableParamError'
    global embedding_function 
    
    try:
        if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
            db = Chroma(persist_directory=DB_PATH, 
                        embedding_function=embedding_function)
            print("✅ Base de datos RAG cargada correctamente.")
            return db
        else:
            print("⚠️ Base de datos no encontrada.")
            return None
    except Exception as e:
        print(f"❌ Error cargando DB: {e}")
        return None

vector_db = load_rag_database()

# --- FUNCIÓN VISUAL: RENDERIZAR PÁGINA PDF ---
def render_pdf_page(filename, page_number):
    """Busca el PDF y convierte la página específica en una imagen."""
    try:
        # Limpiamos el nombre del archivo
        clean_filename = os.path.basename(filename)
        pdf_path = os.path.join(PDF_FOLDER, clean_filename)
        
        if not os.path.exists(pdf_path):
            # Si no encuentra el archivo en la carpeta 'pdfs', no hace nada
            return None
        
        doc = fitz.open(pdf_path)
        # Validar que la página existe
        if 0 <= page_number < len(doc):
            page = doc.load_page(page_number)
            # Renderizar con Zoom x2 para mejor calidad
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
            img_data = pix.tobytes("png")
            return Image.open(io.BytesIO(img_data))
            
    except Exception as e:
        print(f"⚠️ Error renderizando PDF: {e}")
    return None

# --- LÓGICA DE RESPUESTA (CON RAG Y VISUALES) ---

def generate_response_with_visuals(prompt):
    contexto_rag = ""
    sources_found = [] # Lista para guardar tuplas (archivo, página)
    
    # 1. Búsqueda en la Base de Datos
    if vector_db:
        try:
            docs = vector_db.similarity_search(prompt, k=3)
            contexto_rag = "\n\n".join([doc.page_content for doc in docs])
            
            # Extraer fuentes únicas para visualización
            for doc in docs:
                src = doc.metadata.get('source', '')
                page = doc.metadata.get('page', 0)
                if src:
                    sources_found.append((src, page))
                    
        except Exception as e:
            print(f"Error en búsqueda vectorial: {e}")
    
    # 2. Construcción del Prompt
    prompt_con_contexto = f"""
    [CONTEXTO RAG]: {contexto_rag}
    [PREGUNTA]: {prompt}
    """
    
    # 3. Generación de Texto
    try:
        # Asegurar que la sesión de chat existe
        if 'chat_session' not in st.session_state:
            st.session_state.chat_session = client.chats.create(
                model=MODELO,
                config=chat_config
            )
            
        response_obj = st.session_state.chat_session.send_message(prompt_con_contexto)
        response_text = response_obj.text
    except Exception as e:
        response_text = f"⚠️ Error generando respuesta: {str(e)}"
        
    return response_text, sources_found

# --- INTERFAZ DE USUARIO STREAMLIT ---

st.set_page_config(page_title="Genio Visual", page_icon="🦉", layout="wide")

st.title("🦉 Genio: Tu Super Tutor Visual")
st.markdown(f"**Estado RAG:** {'✅ Activo' if vector_db else '⚠️ Inactivo (Solo conocimiento general)'}")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "¡Hola! Soy Genio. Pregúntame lo que quieras y te mostraré de dónde saco la información. 📸"})

# 1. Mostrar mensajes antiguos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Si el mensaje tiene imágenes guardadas, las mostramos
        if "images" in message:
            for img_info in message["images"]:
                with st.expander(f"🔍 Fuente: {img_info['name']} (Pág {img_info['page'] + 1})"):
                    st.image(img_info['image'], use_column_width=True)

# 2. Input de Chat
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta
    with st.spinner('Genio está pensando y buscando en los libros...'):
        response_text, sources = generate_response_with_visuals(prompt)
    
    # Procesar imágenes nuevas (si las hay)
    images_to_save = []
    
    # Mostrar respuesta del asistente
    with st.chat_message("assistant"):
        st.markdown(response_text)
        
        if sources:
            # Filtrar duplicados para no mostrar la misma página 3 veces
            seen = set()
            unique_sources = [x for x in sources if not (x in seen or seen.add(x))]
            
            for src, page_num in unique_sources:
                img = render_pdf_page(src, page_num)
                if img:
                    clean_name = os.path.basename(src)
                    with st.expander(f"📸 Ver página original: {clean_name} (Pág {page_num + 1})"):
                        st.image(img, caption=f"Fuente: {clean_name}", use_column_width=True)
                    
                    # Guardar imagen en memoria para el historial
                    images_to_save.append({'name': clean_name, 'page': page_num, 'image': img})
    
    # Guardar en el historial de sesión
    msg_data = {"role": "assistant", "content": response_text}
    if images_to_save:
        msg_data["images"] = images_to_save
    st.session_state.messages.append(msg_data)
