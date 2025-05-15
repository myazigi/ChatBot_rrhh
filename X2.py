# -*- coding: utf-8 -*-
import streamlit as st
import os
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import traceback
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

# --- Streamlit config: ¡Debe ser lo primero! ---
st.set_page_config(page_title="Aura RRHH (WhatsApp)", page_icon="🤖", layout="centered")

# --- LLM Backend Configuration ---
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemma-3-1b-it-qat")
try:
    name_parts = re.split(r'[:/-]', LLM_MODEL_NAME)
    DISPLAY_MODEL_NAME = name_parts[-1].replace('-', ' ').title() if name_parts else LLM_MODEL_NAME
    if not DISPLAY_MODEL_NAME: DISPLAY_MODEL_NAME = "LLM Local"
except Exception: DISPLAY_MODEL_NAME = "LLM Local"

# --- CSS Estilo WhatsApp ---
CHAT_CONTAINER_ID_FOR_JS = "whatsapp-chat-scroll-area"
CHAT_CONTAINER_CLASS_FOR_CSS = "chat-messages-container-class"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    :root {{
        --whatsapp-bg: #e5ddd5;
        --whatsapp-user-bubble-bg: #dcf8c6;
        --whatsapp-bot-bubble-bg: #ffffff;
        --whatsapp-text-dark: #303030;
        --whatsapp-meta-text: #8b9194;
        --whatsapp-header-bg: #075e54;
        --whatsapp-header-text: #ffffff;
        --whatsapp-input-bg: #f0f0f0;
        --font-family-whatsapp: 'Roboto', sans-serif;
    }}
    html, body {{
        height: 100% !important; margin: 0 !important; padding: 0 !important;
        overflow: hidden !important; font-family: var(--font-family-whatsapp) !important;
        background-color: var(--whatsapp-bg) !important;
    }}
    .block-container {{
        max-width: 800px !important; min-width: 350px !important;
        background-color: var(--whatsapp-bg) !important; padding: 0 !important; margin: 0 auto !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); height: 100vh !important;
        display: flex !important; flex-direction: column !important; overflow: hidden !important;
    }}
    .whatsapp-header {{
        background-color: var(--whatsapp-header-bg); color: var(--whatsapp-header-text);
        padding: 10px 15px; display: flex; align-items: center;
        font-size: 1.1rem; font-weight: 500; flex-shrink: 0;
    }}
    .whatsapp-header img.avatar {{ width: 40px; height: 40px; border-radius: 50%; margin-right: 10px; }}

    .{CHAT_CONTAINER_CLASS_FOR_CSS} {{ /* CONTENEDOR DE MENSAJES CON SCROLL */
        flex-grow: 1 !important; overflow-y: auto !important; min-height: 0;
        padding: 10px; background-color: var(--whatsapp-bg);
        /* Eliminado display:flex y flex-direction:column de aquí */
        /* border: 2px solid yellow !important; /* DEBUG */
    }}

    [data-testid="stChatMessage"] {{ /* Contenedor de cada mensaje (avatar+burbuja) */
        margin-top: 0 !important;
        margin-bottom: 8px !important; /* Espacio entre mensajes */
    }}

    [data-testid="stChatMessageContent"] {{ padding: 7px 12px !important; border-radius: 7.5px !important; box-shadow: 0 1px 0.5px rgba(0,0,0,0.13) !important; width: fit-content !important; max-width: 75% !important; overflow-wrap: break-word !important; word-break: break-word !important; line-height: 1.4 !important; font-size: 0.95rem !important; margin-bottom: 2px; }}
    [data-testid="stChatMessageContent"] p {{ margin-bottom: 0 !important; white-space: pre-line !important; color: var(--whatsapp-text-dark) !important; }}
    [data-testid="stChatMessage"][role="user"] {{ display: flex; flex-direction: column; align-items: flex-end; margin-left: auto; /* margin-bottom ya está en stChatMessage general */ }}
    [data-testid="stChatMessage"][role="user"] [data-testid="stChatMessageContent"] {{ background-color: var(--whatsapp-user-bubble-bg) !important; }}
    [data-testid="stChatMessage"][role="assistant"] {{ display: flex; flex-direction: column; align-items: flex-start; margin-right: auto; /* margin-bottom ya está en stChatMessage general */ }}
    [data-testid="stChatMessage"][role="assistant"] [data-testid="stChatMessageContent"] {{ background-color: var(--whatsapp-bot-bubble-bg) !important; }}
    .whatsapp-meta {{ font-size: 0.75rem; color: var(--whatsapp-meta-text); margin-top: 0px; padding: 0 5px; text-align: right; }}
    [data-testid="stChatMessage"][role="assistant"] .whatsapp-meta {{ text-align: left; }}
    .whatsapp-meta .ticks {{ margin-left: 3px; font-weight: bold; color: #4fc3f7; }}

    .stChatInputContainer {{
        background-color: var(--whatsapp-input-bg) !important;
        border-top: 1px solid #d1d7db !important;
        padding: 8px 10px !important; flex-shrink: 0;
    }}
    .stTextInput input {{ border-radius: 20px !important; border: 1px solid #e0e0e0 !important; padding: 10px 15px !important; font-size: 0.95rem !important; background-color: #ffffff !important; color: var(--whatsapp-text-dark) !important; }}
    .stTextInput input:focus {{ border-color: var(--whatsapp-header-bg) !important; box-shadow: none !important; }}
    header[data-testid="stHeader"] {{ display: none !important; }}
    footer {{ display: none !important; }}
</style>
""", unsafe_allow_html=True)

# --- Simulación de Cabecera de WhatsApp ---
bot_avatar_url = "https://cdn-icons-png.flaticon.com/512/1698/1698535.png"
st.markdown(f""" <div class="whatsapp-header"> <img src="{bot_avatar_url}" class="avatar" alt="Bot Avatar"> Aura RRHH IA </div> """, unsafe_allow_html=True)

# --- Globales y Funciones (SIN CAMBIOS) ---
DOC_LOAD_ERROR = False; VECTORIZER_GLOBAL = None; CHUNKS_GLOBAL = None; CORPUS_GLOBAL = None; CORPUS_VECS_GLOBAL = None
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
pdf_file_path = os.path.join(script_dir, "Prueba4.pdf")
codigo_trabajo_path = os.path.join(script_dir, "Código del Trabajo-Chile.pdf")
CHUNK_SIZE = 600; CHUNK_OVERLAP = 75
if not os.path.exists(pdf_file_path): st.sidebar.error(f"Manual no encontrado"); DOC_LOAD_ERROR = True
if not os.path.exists(codigo_trabajo_path): st.sidebar.error(f"Código del Trabajo no encontrado"); DOC_LOAD_ERROR = True
@st.cache_data(show_spinner="Analizando documentos...")
def load_and_chunk_pdfs(manual_path: str, codigo_path: str) -> List[Dict[str, Any]]:
    all_chunks = []; doc_sources = {"Manual": manual_path, "Código Trabajo": codigo_path}
    for doc_name, path in doc_sources.items():
        if not os.path.exists(path): continue
        try:
            reader = PyPDF2.PdfReader(path)
            for i, page in enumerate(reader.pages):
                pg_text = page.extract_text()
                if not pg_text or not pg_text.strip(): continue
                pg_text = re.sub(r'\s+', ' ', pg_text).strip(); pg_text = re.sub(r'-\n', '', pg_text)
                for start in range(0, len(pg_text), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk_text = pg_text[start : start + CHUNK_SIZE].strip()
                    if len(chunk_text) > 50: all_chunks.append({'fuente': doc_name, 'pagina': i + 1, 'texto': chunk_text})
        except Exception as e: st.sidebar.error(f"Error procesando PDF '{doc_name}'."); print(traceback.format_exc())
    if not all_chunks and not DOC_LOAD_ERROR: st.sidebar.warning("No se pudieron extraer chunks.")
    return all_chunks
@st.cache_data(show_spinner="Creando índice...")
def precalculate_tfidf_vectors(_chunks: List[Dict[str, Any]]) -> Tuple[Optional[List[str]], Optional[TfidfVectorizer], Optional[Any]]:
    if not _chunks: return None, None, None
    try:
        corpus = [c['texto'] for c in _chunks]; vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85, min_df=2)
        corpus_vectors = vectorizer.fit_transform(corpus); return corpus, vectorizer, corpus_vectors
    except Exception as e: st.sidebar.error(f"Error calculando TF-IDF."); print(traceback.format_exc()); return None, None, None
if not DOC_LOAD_ERROR:
    CHUNKS_GLOBAL = load_and_chunk_pdfs(pdf_file_path, codigo_trabajo_path)
    if not CHUNKS_GLOBAL: DOC_LOAD_ERROR = True
    else:
        CORPUS_GLOBAL, VECTORIZER_GLOBAL, CORPUS_VECS_GLOBAL = precalculate_tfidf_vectors(CHUNKS_GLOBAL)
        if CORPUS_VECS_GLOBAL is None: DOC_LOAD_ERROR = True
def find_relevant_context(pregunta: str, top_n: int = 1) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if DOC_LOAD_ERROR or not CHUNKS_GLOBAL or not pregunta.strip() or VECTORIZER_GLOBAL is None or CORPUS_VECS_GLOBAL is None: return None, []
    try:
        pregunta_vec = VECTORIZER_GLOBAL.transform([pregunta]); similitudes = cosine_similarity(pregunta_vec, CORPUS_VECS_GLOBAL)[0]
        relevant_indices_scores = sorted(enumerate(similitudes), key=lambda item: item[1], reverse=True)
        threshold = 0.05; top_indices = [idx for idx, score in relevant_indices_scores if score > threshold][:top_n]
        if not top_indices: return None, []
        contexto_combinado = "Contexto Relevante:\n\n"; relevant_chunks_data_dummy = []; seen_texts = set()
        for idx in top_indices:
            chunk_data = CHUNKS_GLOBAL[idx]
            if chunk_data['texto'] not in seen_texts:
                contexto_combinado += f"Fuente: {chunk_data['fuente']}, Pág. ~{chunk_data['pagina']}\n'{chunk_data['texto']}'\n\n"; seen_texts.add(chunk_data['texto'])
        return contexto_combinado.strip(), relevant_chunks_data_dummy
    except Exception as e: print(f"Error en búsqueda: {e}"); print(traceback.format_exc()); return None, []
def get_llm_response_from_api(system_prompt: str, user_prompt: str, model_name: str, api_url: str) -> str:
    headers = {'Content-Type': 'application/json'}; payload = { "model": model_name, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.4, "max_tokens": 1024, "stream": False }
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=60) 
        if response.status_code == 404: return f"Error 404: Servidor LLM no encontrado."
        if response.status_code == 400:
            error_detail_json = {}; error_detail_text = response.text 
            try: error_detail_json = response.json(); error_message_from_json = error_detail_json.get('error', {}).get('message');
            if error_message_from_json: error_detail_text = error_message_from_json
            except json.JSONDecodeError: pass 
            return f"Error 400: Petición LLM inválida. {str(error_detail_text)[:200]}"
        response.raise_for_status(); response_data = response.json()
        if 'choices' in response_data and response_data['choices']: content = response_data['choices'][0].get('message', {}).get('content', ''); return content.strip() if content else "Recibí una respuesta vacía."
        return f"Respuesta LLM inesperada."
    except requests.exceptions.Timeout: return "Error: Timeout LLM."
    except requests.exceptions.ConnectionError: return f"Error: Conexión LLM."
    except requests.exceptions.RequestException as e: return f"Error req LLM: {type(e).__name__}"
    except json.JSONDecodeError: return f"Error: JSON LLM."
    except Exception as e: print(traceback.format_exc()); return f"Error LLM: {type(e).__name__}"
def handle_new_message(user_prompt: str):
    current_time = datetime.now().strftime("%H:%M")
    st.session_state.chat.append({"role": "user", "content": user_prompt, "timestamp": current_time})
    manual_filename = os.path.basename(pdf_file_path); codigo_filename = os.path.basename(codigo_trabajo_path)
    system_prompt = f"""Eres Aura, asistente RRHH conciso. Docs: '{manual_filename}', '{codigo_filename}'.
    Instrucciones: Corto y directo. Basado ESTRICTAMENTE en contexto. Si no hay contexto, indica que la info no está. Tono profesional, cercano. Si pregunta ajena, indica que solo ayudas con docs."""
    contexto, _ = find_relevant_context(user_prompt, top_n=1); final_user_prompt = ""
    if contexto: max_context_len = 1500; final_user_prompt = f"Contexto:\n{contexto[:max_context_len]}\n\nPregunta: {user_prompt}\nResponde brevemente."
    else: final_user_prompt = f"Pregunta: {user_prompt}\nNo hay contexto. Indica que la info no está en los docs."
    llm_response_raw = get_llm_response_from_api(system_prompt, final_user_prompt, LLM_MODEL_NAME, LLM_API_URL)
    bot_current_time = datetime.now().strftime("%H:%M")
    if "Error:" in llm_response_raw or "Error " in llm_response_raw: st.session_state.chat.append({'role': 'assistant', 'content': f"⚠️ Problema técnico: {llm_response_raw}", 'timestamp': bot_current_time})
    else: st.session_state.chat.append({'role': 'assistant', 'content': llm_response_raw if llm_response_raw else "No pude responder.", 'timestamp': bot_current_time})
# --- === App Execution Starts Here === ---
if "chat" not in st.session_state: st.session_state.chat = []
if not st.session_state.chat:
    initial_bot_message = '¡Hola! Soy Aura ✨. ¿Consultas sobre el Manual o Código del Trabajo?'
    if DOC_LOAD_ERROR: initial_bot_message = f'⚠️ No pude cargar los docs. Mis respuestas pueden ser limitadas.'
    elif CHUNKS_GLOBAL is None or VECTORIZER_GLOBAL is None and not DOC_LOAD_ERROR: initial_bot_message = f'⚠️ Problema al preparar datos.'
    st.session_state.chat.append({'role': 'assistant', 'content': initial_bot_message, 'timestamp': datetime.now().strftime("%H:%M")})

# --- Contenedor explícito para los mensajes de chat ---
st.markdown(f"<div id='{CHAT_CONTAINER_ID_FOR_JS}' class='{CHAT_CONTAINER_CLASS_FOR_CSS}'>", unsafe_allow_html=True)
for message in st.session_state.chat:
    role = message['role']
    with st.chat_message(name=role, avatar=None):
        st.markdown(message.get('content', '...'))
        meta_html_parts = [f"<span class='timestamp'>{message.get('timestamp', '')}</span>"]
        if role == "user": meta_html_parts.append("<span class='ticks'>✓✓</span>")
        st.markdown(f"<div class='whatsapp-meta'>{''.join(meta_html_parts)}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Chat Input Widget ---
def submit_prompt_callback():
    prompt_value = st.session_state.whatsapp_chat_input_key_v6 
    if prompt_value and prompt_value.strip():
        handle_new_message(prompt_value)
st.chat_input( "Escribe un mensaje...", key="whatsapp_chat_input_key_v6", on_submit=submit_prompt_callback, disabled=DOC_LOAD_ERROR )

# --- JavaScript para auto-scroll ---
scroll_script = f"""
<script>
    function attemptScrollToBottom() {{
        const chatContainer = document.getElementById('{CHAT_CONTAINER_ID_FOR_JS}');
        if (chatContainer) {{
            requestAnimationFrame(() => {{
                chatContainer.scrollTop = chatContainer.scrollHeight;
                // console.log('JS Scroll: scrollTop set to', chatContainer.scrollHeight, 'for', chatContainer.id); // DEBUG
            }});
        }} else {{ /* console.log('JS Scroll: Container "{CHAT_CONTAINER_ID_FOR_JS}" not found.'); /* DEBUG */ }}
    }}
    attemptScrollToBottom(); // Ejecutar en cada rerun
</script>
"""
if st.session_state.chat: st.components.v1.html(scroll_script, height=1)
if DOC_LOAD_ERROR and len(st.session_state.get("chat", [])) <= 1 : st.toast("⚠️ Chat deshabilitado.", icon="🚫")