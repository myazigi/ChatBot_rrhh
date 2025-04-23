import streamlit as st
import os
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summa import summarizer

# --- Streamlit config: ¡Debe ser lo primero! ---
st.set_page_config(page_title="Chatbot RRHH", page_icon=":busts_in_silhouette:", layout="centered")

# --- CSS --- 
st.markdown("""
<style>
.chat-msg-user {
    background: linear-gradient(90deg, #d0ebff 0%, #b2f2ff 100%);
    border-radius: 16px 16px 6px 16px;
    padding: 12px 16px;
    margin: 8px 0 8px 25%;
    max-width: 75%;
    text-align: right;
    font-size: 17px;
    box-shadow: 1px 2px 5px #bbe2fa39;
}
.chat-msg-bot {
    background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 16px 16px 16px 6px;
    padding: 12px 16px;
    margin: 8px 25% 8px 0;
    max-width: 75%;
    text-align: left;
    font-size: 17px;
    box-shadow: 1px 2px 5px #c9ced639;
}
.chat-ref {
    color: #889;
    font-size: 0.97em;
    margin-top: 5px;
    display: block;
}
details {
    margin-top:8px;
    margin-bottom:2px;
    font-size:0.97em;
}
summary {
    cursor: pointer;
    color: #227;
}
</style>
""", unsafe_allow_html=True)

# --- Funciones PDF / AI ---

@st.cache_data(show_spinner="Procesando PDF...")
def load_pdf_and_chunks(pdf_path):
    if not os.path.exists(pdf_path):
        return []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        chunks = []
        for i, page in enumerate(reader.pages):
            pg_text = page.extract_text() or ""
            for par in re.split(r'\n{2,}', pg_text):
                texto = par.strip().replace('\n', ' ')
                if len(texto) > 40:
                    chunks.append({'pagina': i+1, 'texto': texto})
    return chunks

def buscar_respuesta(pregunta, chunks):
    if not chunks:
        return "No hay contenido disponible.", 0, "Sin resumen"
    corpus = [c['texto'] for c in chunks]
    vectorizer = TfidfVectorizer().fit(corpus + [pregunta])
    corpus_vecs = vectorizer.transform(corpus)
    pregunta_vec = vectorizer.transform([pregunta])
    similitudes = cosine_similarity(pregunta_vec, corpus_vecs)[0]
    idx = similitudes.argmax()
    parrafo = chunks[idx]

    # ---- Generar resumen ejecutivo ----
    resumen = summarizer.summarize(parrafo['texto'], words=60, language='spanish')
    if not resumen:
        resumen = parrafo['texto'].split('. ')[0] + '.'

    return parrafo['texto'], parrafo['pagina'], resumen

# --- Título y descripción ---
st.markdown("<h2 style='text-align:center'>🤖 Chatbot de Recursos Humanos</h2>", unsafe_allow_html=True)
st.write("Ingrese su consulta sobre RRHH basada en el archivo **Prueba4.pdf**. El bot responde de forma ejecutiva y cita la fuente.")

# --- Cargar el PDF ---
PDF_PATH = "Prueba4.pdf"
chunks = load_pdf_and_chunks(PDF_PATH)
if not chunks:
    st.warning("No se encontró o no se pudo leer el archivo Prueba4.pdf. Colócalo en el mismo directorio.")
    st.stop()

# --- Estado del chat ---
if "chat" not in st.session_state:
    st.session_state["chat"] = []

# --- Procesamiento del nuevo mensaje ---
def procesar_mensaje():
    user_question = st.session_state.input_text
    if not user_question or not user_question.strip():
        return
    st.session_state.chat.append({'role': 'user', 'msg': user_question.strip()})
    texto_ori, pagina, resumen = buscar_respuesta(user_question, chunks)
    st.session_state.chat.append({'role': 'bot', 'msg': texto_ori, 'ref': pagina, 'resumen': resumen})
    st.session_state.input_text = ""  # Limpiar

# --- Mostrar el chat (profesional, moderno) ---
for turno in st.session_state["chat"]:
    if turno['role'] == 'user':
        st.markdown(
            f'<div class="chat-msg-user"><b>🙋‍♂️ Tú</b><br>{turno["msg"]}</div>',
            unsafe_allow_html=True)
    elif turno['role'] == 'bot':
        # Profesional: resumen ejecutivo, original expandible, cita
        st.markdown(
            f'''
            <div class="chat-msg-bot">
              <b>🤖 RRHHBot (Resumen Ejecutivo)</b><br>
              <b>Resumen:</b> {turno["resumen"]}<br>
              <details>
                  <summary>Ver texto original</summary>
                  {turno["msg"]}
              </details>
              <div class="chat-ref">Fuente: Prueba4.pdf, página {turno["ref"]}</div>
            </div>
            ''', unsafe_allow_html=True)

# --- Formulario siempre al final (abajo) ---
with st.form(key="form_chat", clear_on_submit=True):
    pregunta = st.text_area(
        "Consulta a RRHH:",
        key="input_text", height=100, placeholder="Ej: ¿Cómo se calculan los días de vacaciones?"
    )
    enviado = st.form_submit_button("Enviar", on_click=procesar_mensaje)

st.caption("Chatbot RRHH: responde con información resumida y ejecutiva, citando la referencia de **Prueba4.pdf**.")

# --- Mistral LLM Backend Configuration ---
# Adapt this URL if your Mistral API endpoint is different (e.g., /v1/chat/completions)
MISTRAL_API_URL = "http://localhost:10001" # Common Ollama endpoint
# Replace with the actual model name served by your local instance (e.g., "mistral", "mistral:7b")
MISTRAL_MODEL_NAME = "gemma3:latest" # <<< CHANGE THIS IF NEEDED

