# -*- coding: utf-8 -*-
import os
import re
import json
import traceback
import requests
import PyPDF2
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuración de la página ---
st.set_page_config(page_title="Aura RRHH (WhatsApp)", page_icon="", layout="centered")

# --- Variables de backend LLM ---
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemma-3-1b-it-qat")
try:
    name_parts = re.split(r'[:/-]', LLM_MODEL_NAME)
    DISPLAY_MODEL_NAME = name_parts[-1].replace('-', ' ').title()
except Exception:
    DISPLAY_MODEL_NAME = LLM_MODEL_NAME

# --- Rutas de los PDFs ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
pdf_file_path = os.path.join(script_dir, "Prueba4.pdf")
codigo_trabajo_path = os.path.join(script_dir, "Código del Trabajo-Chile.pdf")

# --- Flags y caches ---
DOC_LOAD_ERROR = False
CHUNKS_GLOBAL = None
CORPUS_GLOBAL = None
VECTORIZER_GLOBAL = None
CORPUS_VECS_GLOBAL = None

# Revisar existencia de los PDFs
if not os.path.exists(pdf_file_path):
    st.sidebar.error("Manual no encontrado")
    DOC_LOAD_ERROR = True
if not os.path.exists(codigo_trabajo_path):
    st.sidebar.error("Código del Trabajo no encontrado")
    DOC_LOAD_ERROR = True

CHUNK_SIZE = 600
CHUNK_OVERLAP = 75

@st.cache_data(show_spinner="Analizando documentos…")
def load_and_chunk_pdfs(manual_path: str, codigo_path: str) -> List[Dict[str, Any]]:
    all_chunks = []
    for nombre, path in [("Manual", manual_path), ("CódigoTrabajo", codigo_path)]:
        if not os.path.exists(path):
            continue
        try:
            reader = PyPDF2.PdfReader(path)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    continue
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'-\n', '', text)
                for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk = text[start:start + CHUNK_SIZE].strip()
                    if len(chunk) > 50:
                        all_chunks.append({
                            "fuente": nombre,
                            "pagina": i + 1,
                            "texto": chunk
                        })
        except Exception:
            st.sidebar.error(f"Error procesando PDF {nombre}")
            print(traceback.format_exc())
    if not all_chunks and not DOC_LOAD_ERROR:
        st.sidebar.warning("No se extrajeron chunks de texto.")
    return all_chunks

@st.cache_data(show_spinner="Creando índice…")
def precalculate_tfidf_vectors(chunks: List[Dict[str, Any]]):
    if not chunks:
        return None, None, None
    try:
        corpus = [c["texto"] for c in chunks]
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2
        )
        corpus_vecs = vectorizer.fit_transform(corpus)
        return corpus, vectorizer, corpus_vecs
    except Exception:
        st.sidebar.error("Error calculando TF-IDF")
        print(traceback.format_exc())
        return None, None, None

def find_relevant_context(pregunta: str, top_n: int = 1) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if DOC_LOAD_ERROR or not CHUNKS_GLOBAL or not pregunta.strip():
        return None, []
    try:
        q_vec = VECTORIZER_GLOBAL.transform([pregunta])
        sims = cosine_similarity(q_vec, CORPUS_VECS_GLOBAL)[0]
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        threshold = 0.05
        top_idxs = [i for i, s in ranked if s > threshold][:top_n]
        if not top_idxs:
            return None, []
        ctx = "Contexto relevante:\n\n"
        seen = set()
        for idx in top_idxs:
            chunk = CHUNKS_GLOBAL[idx]
            if chunk["texto"] not in seen:
                ctx += f"Fuente: {chunk['fuente']}, Pág. ~{chunk['pagina']}\n"
                ctx += f'"{chunk["texto"]}"\n\n'
                seen.add(chunk["texto"])
        return ctx.strip(), []
    except Exception:
        print(traceback.format_exc())
        return None, []

def get_llm_response(system_prompt: str, user_prompt: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 1024,
        "stream": False
    }
    try:
        r = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=60)
        if r.status_code == 404:
            return "Error 404: Servidor LLM no encontrado."
        if r.status_code == 400:
            detail = r.json().get("error", {}).get("message", r.text)
            return f"Error 400: petición inválida. {detail}"
        r.raise_for_status()
        data = r.json()
        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        return content.strip() or "Recibí una respuesta vacía."
    except requests.exceptions.Timeout:
        return "Error: Timeout LLM."
    except requests.exceptions.ConnectionError:
        return "Error: Conexión LLM."
    except Exception as e:
        print(traceback.format_exc())
        return f"Error LLM: {type(e).__name__}"

def handle_new_message(user_text: str):
    ts = datetime.now().strftime("%H:%M")
    if "chat" not in st.session_state:
        st.session_state.chat = []
    st.session_state.chat.append({
        "role": "user",
        "content": user_text,
        "timestamp": ts
    })
    sys_prompt = (
        f"Eres Aura, asistente RRHH conciso. Docs: "
        f"'{os.path.basename(pdf_file_path)}', "
        f"'{os.path.basename(codigo_trabajo_path)}'.\n"
        "Si no hay contexto, di que la info no está. Tono profesional y cercano."
    )
    ctx, _ = find_relevant_context(user_text, top_n=1)
    if ctx:
        final_user = f"Contexto:\n{ctx[:1500]}\n\nPregunta: {user_text}"
    else:
        final_user = f"Pregunta: {user_text}\nNo hay contexto."
    llm_resp = get_llm_response(sys_prompt, final_user)
    bot_ts = datetime.now().strftime("%H:%M")
    if llm_resp.startswith("Error"):
        content = f"⚠️ Problema técnico: {llm_resp}"
    else:
        content = llm_resp
    st.session_state.chat.append({
        "role": "assistant",
        "content": content,
        "timestamp": bot_ts
    })

# --- Carga de PDFs e índices ---
if not DOC_LOAD_ERROR:
    CHUNKS_GLOBAL = load_and_chunk_pdfs(pdf_file_path, codigo_trabajo_path)
    if CHUNKS_GLOBAL:
        CORPUS_GLOBAL, VECTORIZER_GLOBAL, CORPUS_VECS_GLOBAL = precalculate_tfidf_vectors(CHUNKS_GLOBAL)
        if CORPUS_VECS_GLOBAL is None:
            DOC_LOAD_ERROR = True
    else:
        DOC_LOAD_ERROR = True

# --- CSS estilo WhatsApp moderno ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    body, html {
        margin: 0;
        padding: 0;
        background: linear-gradient(135deg, #ece5dd 0%, #d1f7c4 100%) !important;
        font-family: 'Roboto', sans-serif;
    }
    .wa-header {
        position: sticky;
        top: 0;
        width: 100%;
        background: var(--header-bg, #075e54);
        color: var(--header-text, #fff);
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.3rem 1rem 1rem;
        border-radius: 0 0 18px 18px;
        z-index: 100;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .wa-avatar {
        width: 44px; height: 44px; border-radius: 50%; background: #fff; display: flex; align-items: center; justify-content: center; font-size: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .wa-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: .5px;
    }
    .chat-container {
        background: rgba(255,255,255,0.85);
        padding: 1.2rem 0.7rem 5.5rem 0.7rem;
        border-radius: 18px;
        min-height: 500px;
        max-width: 540px;
        margin: 0 auto 0 auto;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }
    .msg-user {
        background: var(--user-bg, #dcf8c6);
        color: var(--text, #303030);
        padding: .7rem 1.1rem;
        border-radius: 10px 10px 0 18px;
        margin-bottom: 4px;
        align-self: flex-end;
        max-width: 80%;
        font-size: 1.04rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        word-break: break-word;
    }
    .msg-bot {
        background: var(--bot-bg, #fff);
        color: var(--text, #303030);
        padding: .7rem 1.1rem;
        border-radius: 10px 10px 18px 0;
        margin-bottom: 4px;
        align-self: flex-start;
        max-width: 80%;
        border: 1px solid #e0e0e0;
        font-size: 1.04rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        word-break: break-word;
    }
    .msg-meta {
        font-size: .75em;
        color: var(--meta, #8b9194);
        margin-top: 0;
        margin-bottom: 2px;
        padding-left: 2px;
    }
    .wa-input-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100vw;
        background: #f7f7f7;
        box-shadow: 0 -1px 8px rgba(0,0,0,0.04);
        padding: 1rem 0;
        z-index: 101;
        display: flex;
        justify-content: center;
    }
    .wa-input-inner {
        display: flex;
        gap: 0.5rem;
        width: 100%;
        max-width: 520px;
    }
    .wa-input-box {
        flex: 1 1 auto;
        border: 1.5px solid #d1d7db;
        border-radius: 12px;
        padding: .7rem 1rem;
        font-size: 1.08rem;
        outline: none;
        background: #fff;
        transition: border 0.2s;
    }
    .wa-input-box:focus {
        border: 1.5px solid #25d366;
    }
    .wa-send-btn {
        background: #25d366;
        color: #fff;
        border: none;
        border-radius: 12px;
        padding: .7rem 1.3rem;
        font-size: 1.09rem;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.2s;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .wa-send-btn:hover {
        background: #128c7e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Interfaz de usuario del chat moderno ---

# Inicializa la sesión de chat si no existe
if "chat" not in st.session_state:
    st.session_state.chat = []


# No mostrar ningún título ni caja de texto fuera del chat

if DOC_LOAD_ERROR:
    st.error("No se pudieron cargar los documentos PDF requeridos. Por favor verifica los archivos.")
else:
        # Título moderno del chatbot
    st.markdown('''
        <div style="display:flex;align-items:center;gap:12px;padding:12px 0 18px 0;">
            <div style="font-size:2rem; background:linear-gradient(135deg,#25d366,#128c7e); border-radius:50%; width:48px; height:48px; display:flex; align-items:center; justify-content:center; color:white;">🤖</div>
            <div style="font-size:1.7rem; font-weight:700; color:#128c7e; letter-spacing:1px;">Aura RRHH</div>
        </div>
        <hr style="margin-bottom:14px; border: none; border-top: 2px solid #e0e0e0;" />
    ''', unsafe_allow_html=True)

    # Mostrar mensajes en orden natural (más antiguo arriba, nuevo abajo)
    if st.session_state.chat:
        for msg in st.session_state.chat:
            if msg["role"] == "user":
                st.markdown(f'<div class="msg-meta">Tú | {msg["timestamp"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="msg-meta">Aura | {msg["timestamp"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        # Mostrar mensaje de bienvenida si el chat está vacío
        st.markdown(f'<div class="msg-meta">Aura | ahora</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="msg-bot">¡Hola! 👋 Soy Aura, tu asistente RRHH. ¿En qué puedo ayudarte hoy?</div>', unsafe_allow_html=True)

    # Entrada de texto y botón para enviar
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Mensaje", "", key="input_text", placeholder="Escribe tu mensaje...", label_visibility="collapsed")
        submit = st.form_submit_button("Enviar")
    if submit and user_input.strip():
        handle_new_message(user_input.strip())
        st.rerun()