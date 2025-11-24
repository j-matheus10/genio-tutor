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
# --- NUEVO IMPORT DE SEGURIDAD ---
from google.genai.types import HarmCategory, HarmBlockThreshold

# --- IMPORTS DE LANGCHAIN ---
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURACIÓN DE PÁGINA (Debe ir al inicio) ---
st.set_page_config(page_title="Tilines Inc.", page_icon="🦉", layout="wide")

# --- CONFIGURACIÓN GENERAL ---
MODELO = "gemini-2.5-flash" 
DB_PATH = "genio_db_knowledge"  
ZIP_PATH = "genio_db_knowledge.zip"
PDF_FOLDER = "pdfs"             
EMBEDDING_MODEL_NAME = "text-embedding-004"

# Asegurar que exista la carpeta de PDFs
os.makedirs(PDF_FOLDER, exist_ok=True)

# --- DESCOMPRESIÓN AUTOMÁTICA ---
if not os.path.exists(DB_PATH) and os.path.exists(ZIP_PATH):
    print("📦 ZIP detectado. Descomprimiendo...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DB_PATH)

# --- CONFIGURACIÓN DE SEGURIDAD Y PERSONALIDAD (FILTROS ÉTICOS) ---

# 1. Filtros de Seguridad Estrictos (Protección Infantil)
# Bloquea contenido de odio, acoso o sexualmente explícito.
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

# 2. System Prompt: Definición de rol y límites pedagógicos
SYSTEM_INSTRUCTION = """
Eres "Tilin", un asistente de inteligencia artificial diseñado para ayudar a niños a estudiar.
IMPORTANTE: NO eres un humano, ni un amigo real, ni un profesor; eres una herramienta digital de apoyo.

Tus Principios Rectores:
1. SEGURIDAD Y SALUD: Jamás toleres bullying, violencia o autolesiones. PROHIBIDO hablar, mencionar o dar ejemplos sobre ALCOHOL, DROGAS o sustancias ilícitas bajo ninguna circunstancia. Además de temas sexuales más allá de libros de biología o consultas básicas sobre funcionamiento.
2. NO REEMPLAZO: Si el niño parece frustrado, triste o el tema es muy complejo, sugiérele amablemente pedir ayuda a sus padres o profesores reales.
3. FOMENTO DEL PENSAMIENTO: Tu objetivo es que el niño piense. Nunca hagas la tarea completa por él; guíalo. En caso de que se complique mucho, ve resolviendo con el paso a paso.
4. EQUIDAD: Si en los textos encuentras estereotipos o sesgos, ignóralos y responde con neutralidad e inclusión.
5. CLARIDAD: Usa explicaciones cortas, amables y sin tecnicismos difíciles.
"""


chat_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.5,        # Menor temperatura para evitar alucinaciones
    max_output_tokens=700,  # Respuestas más cortas para evitar sobreestimulación
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

# --- FUNCIONES DE APRENDIZAJE EN VIVO ---
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

# --- FUNCIONES EXTRA ---
def get_rag_sources():
    if not st.session_state.vector_db: return []
    try:
        data = st.session_state.vector_db.get()
        metadatas = data.get('metadatas', [])
        unique = set([os.path.basename(m['source']) for m in metadatas if m and 'source' in m])
        return list(unique)
    except: return []

# --- LÓGICA DE RESPUESTA INTELIGENTE Y ÉTICA ---
def generate_response(prompt, mode):
    contexto_rag = ""
    instruccion_modo = ""
    usar_rag = True

    # 1. Configuración del MODO (Diferenciación estricta)
    if mode == "Enseñar (Socrático)":
        instruccion_modo = """
        MODO SOCRÁTICO PURO:
        - TU OBJETIVO: Que el niño llegue solo a la respuesta.
        - RESTRICCIÓN: NO expliques el tema todavía.
        - ACCIÓN: Responde ÚNICAMENTE con una pregunta que le haga reflexionar sobre su error o sobre el siguiente paso lógico.
        - Ejemplo: Si pregunta "¿Cuánto es 5x5?", tú respondes: "¿Recuerdas qué significa multiplicar? ¿Qué pasa si sumas 5 veces el número 5?".
        """
    elif mode == "Guía Estructurada": 
        instruccion_modo = """
        MODO PROFESOR EXPLICATIVO:
        - TU OBJETIVO: Que el niño entienda el PROCEDIMIENTO o la TEORÍA.
        - ACCIÓN: Explica paso a paso la lógica para resolver este tipo de problemas.
        - RESTRICCIÓN: NO des la solución final (el número o palabra exacta).
        - Ejemplo: Si pregunta "¿Cuánto es 5x5?", tú respondes: "La multiplicación es una suma repetida. Para resolver 5x5, debes sumar el número 5 un total de cinco veces. Intenta hacer esa suma paso a paso.".
        """
    elif mode == "Conocimiento General":
        instruccion_modo = """
        MODO ASISTENTE GENERAL:
        - Responde usando conocimiento general verificado. Y explicando los porqués o paso a paso según el caso.
        - Asegúrate de que el tono sea apto para niños.
        - Si el tema toca alcohol o drogas, niégate a responder y cambia de tema amablemente.
        """
        usar_rag = False 

    # 2. Búsqueda RAG
    if usar_rag and st.session_state.vector_db:
        try:
            docs = st.session_state.vector_db.similarity_search(prompt, k=3)
            contexto_rag = "\n\n".join([doc.page_content for doc in docs])
        except: pass

    # 3. Construcción del Prompt con FILTRO DE SESGOS Y TEMAS PROHIBIDOS
    if usar_rag:
        prompt_ctx = f"""
        {instruccion_modo}
        
        INSTRUCCIÓN DE SEGURIDAD ADICIONAL: Si el contexto o la pregunta menciona alcohol, drogas o vicios, IGNÓRALO y di que no puedes hablar de eso.
        INSTRUCCIÓN DE EQUIDAD: Si el siguiente contexto contiene estereotipos, ignóralos.
        
        CONTEXTO DE LOS LIBROS:
        ---
        {contexto_rag}
        ---
        
        PREGUNTA DEL ESTUDIANTE: {prompt}
        """
    else:
        prompt_ctx = f"""
        {instruccion_modo}
        
        INSTRUCCIÓN DE SEGURIDAD: Si la pregunta menciona alcohol o drogas, responde: "Lo siento, no hablo sobre esos temas. ¿Podemos estudiar otra cosa?".
        
        PREGUNTA DEL ESTUDIANTE: {prompt}
        """

    try:
        if 'chat_session' not in st.session_state:
            st.session_state.chat_session = client.chats.create(model=MODELO, config=chat_config)
        return st.session_state.chat_session.send_message(prompt_ctx).text
    except Exception as e: return f"Lo siento, no puedo responder a eso. (Error: {e})"


# --- INTERFAZ DE USUARIO ---

# --- BARRA LATERAL (CONTROLES) ---
with st.sidebar:
    st.header("🎛️ Configuración")

    # SELECTOR DE MODO ACTUALIZADO
    modo_seleccionado = st.radio(
        "Elige tu modo de estudio:",
        ["Enseñar (Socrático)", "Guía Estructurada", "Conocimiento General"],
        captions=["Te hace pensar con preguntas.", "Te explica el proceso (sin dar la respuesta final).", "Ayuda general sin usar tus libros."]
    )

    st.divider()

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
        st.success(f"✅ Memoria RAG Activa")
        with st.expander("Ver libros indexados"):
            for s in get_rag_sources(): st.markdown(f"- 📄 `{s}`")
    else:
        st.warning("⚠️ RAG Inactivo (No hay PDFs)")

# --- ÁREA PRINCIPAL ---
st.title("🦉 Tilines Inc: IA para estudio")

# Mensaje de bienvenida
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¡Hola! Soy Tilin. Una herramienta de estudio, escoge el modo en que prefieres estudiar y empecemos a aprender juntos. 📚"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.spinner('Pensando...'):
        response_text = generate_response(prompt, modo_seleccionado)

    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
