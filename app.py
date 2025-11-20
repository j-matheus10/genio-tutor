# app.py
# --- FIX PARA SQLITE EN STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- IMPORTS ---
import os
import streamlit as st
import fitz  # PyMuPDF
import io
from PIL import Image
from google import genai
from google.genai import types
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings 

# --- CONFIGURACIÓN DE VARIABLES GLOBALES ---
MODELO = "gemini-2.5-flash" 
DB_PATH = "genio_db_knowledge"  
PDF_FOLDER = "pdfs"             
EMBEDDING_MODEL_NAME = "text-embedding-004"

# --- PERSONALIDAD DEL TUTOR ---
SYSTEM_INSTRUCTION = """
Eres "Genio", un tutor socrático y asistente resolutivo.
1. MODO ENSEÑAR (Predeterminado): NUNCA dar la respuesta directa. Usar preguntas guía.
2. MODO RESOLVER (Guía Resolutivo): Si se pide explícitamente, dar la respuesta directa.
- Usa **negritas** para destacar palabras clave.
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
        model=EMBEDDING_MODEL_NAME, google_api_key=api_key
    )
    return client, embedding_function

client, embedding_function = initialize_gemini()

@st.cache_resource 
def load_rag_database(): 
    global embedding_function 
    try:
        if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
            db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
            return db
        else:
            return None
    except Exception as e:
        return None

vector_db = load_rag_database()

# --- FUNCIÓN RECUPERADA: LISTAR ARCHIVOS ---
def get_rag_sources():
    """Obtiene la lista de archivos PDF indexados en la base de datos."""
    if not vector_db:
        return []
    try:
        data = vector_db.get()
        metadatas = data.get('metadatas', [])
        unique_sources = set()
        for m in metadatas:
            if m and 'source' in m:
                unique_sources.add(os.path.basename(m['source']))
        return list(unique_sources)
    except Exception:
        return []

# --- FUNCIÓN VISUAL: RENDERIZAR PDF ---
def render_pdf_page(filename, page_number):
    try:
        clean_filename = os.path.basename(filename)
        pdf_path = os.path.join(PDF_FOLDER, clean_filename)
        if not os.path.exists(pdf_path): return None
        
        doc = fitz.open(pdf_path)
        if 0 <= page_number < len(doc):
            page = doc.load_page(page_number)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
            img_data = pix.tobytes("png")
            return Image.open(io.BytesIO(img_data))
    except Exception:
        pass
    return None

# --- LÓGICA DE RESPUESTA ---
def generate_response_with_visuals(prompt):
    contexto_rag = ""
    sources_found = [] 
    
    if vector_db:
        try:
            docs = vector_db.similarity_search(prompt, k=3)
            contexto_rag = "\n\n".join([doc.page_content for doc in docs])
            for doc in docs:
                src = doc.metadata.get('source', '')
                page = doc.metadata.get('page', 0)
                if src: sources_found.append((src, page))
        except Exception: pass
    
    prompt_con_contexto = f"[CONTEXTO RAG]: {contexto_rag}\n[PREGUNTA]: {prompt}"
    
    try:
        if 'chat_session' not in st.session_state:
            st.session_state.chat_session = client.chats.create(model=MODELO, config=chat_config)
        response_obj = st.session_state.chat_session.send_message(prompt_con_contexto)
        return response_obj.text, sources_found
    except Exception as e:
        return f"⚠️ Error: {str(e)}", []

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="Genio Tutor", page_icon="🦉", layout="wide")

# === BARRA LATERAL (SIDEBAR) RECUPERADA ===
with st.sidebar:
    st.header("📂 Biblioteca de Genio")
    st.markdown("---")
    if vector_db:
        st.success(f"✅ Memoria Activa")
        sources = get_rag_sources()
        if sources:
            st.markdown(f"**📚 {len(sources)} Libros Indexados:**")
            for source in sources:
                st.markdown(f"- 📄 `{source}`")
        else:
            st.caption("No se encontraron nombres de archivo.")
    else:
        st.error("❌ RAG Inactivo")
        st.warning("Asegúrate de subir la carpeta 'genio_db_knowledge' a GitHub.")
    
    st.markdown("---")
    st.info("💡 **Tip:** Si preguntas algo sobre estos libros, Genio te mostrará la página.")

# === ÁREA PRINCIPAL ===
st.title("🦉 Genio: Tu Super Tutor Visual")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy Genio. Pregúntame lo que quieras y te mostraré de dónde saco la información. 📸"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "images" in message:
            for img_info in message["images"]:
                with st.expander(f"🔍 Fuente: {img_info['name']} (Pág {img_info['page'] + 1})"):
                    st.image(img_info['image'], use_column_width=True)

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('Buscando en la biblioteca...'):
        response_text, sources = generate_response_with_visuals(prompt)
    
    images_to_save = []
    with st.chat_message("assistant"):
        st.markdown(response_text)
        if sources:
            seen = set()
            unique_sources = [x for x in sources if not (x in seen or seen.add(x))]
            for src, page_num in unique_sources:
                img = render_pdf_page(src, page_num)
                if img:
                    clean_name = os.path.basename(src)
                    with st.expander(f"📸 Ver página original: {clean_name} (Pág {page_num + 1})"):
                        st.image(img, caption=f"Fuente: {clean_name}", use_column_width=True)
                    images_to_save.append({'name': clean_name, 'page': page_num, 'image': img})
    
    msg_data = {"role": "assistant", "content": response_text}
    if images_to_save: msg_data["images"] = images_to_save
    st.session_state.messages.append(msg_data)
        msg_data["images"] = images_to_save
    st.session_state.messages.append(msg_data)

