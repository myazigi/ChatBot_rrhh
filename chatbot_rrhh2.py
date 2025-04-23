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

# --- Streamlit config: ¡Debe ser lo primero! ---
st.set_page_config(page_title="Chatbot RRHH IA", page_icon="💼", layout="wide")

# --- LLM Backend Configuration ---
LLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL_NAME = "gemma-3-4b-it-qat" # ¡VERIFICA ESTO!

# --- Crear Nombre Corto para Mostrar ---
try:
    name_parts = re.split(r'[:/-]', LLM_MODEL_NAME)
    DISPLAY_MODEL_NAME = name_parts[0].capitalize() if name_parts else LLM_MODEL_NAME
    if not DISPLAY_MODEL_NAME: DISPLAY_MODEL_NAME = "LLM Local"
except Exception: DISPLAY_MODEL_NAME = "LLM Local"

# --- CSS Modernizado y Mejorado ---
st.markdown(f"""
<style>
    body, .main {{
        background: linear-gradient(135deg, #e3fdfd 0%, #cbf1f5 50%, #a6e3e9 100%) !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    }}
    .block-container {{
        max-width: 900px;
        margin: 0 auto;
        background: #e3fdfdcc;
        border-radius: 22px;
        box-shadow: 0 8px 32px 0 rgba(166, 227, 233, 0.13);
        padding: 2rem 2.5rem 2rem 2.5rem;
    }}
    .stChatMessage {{ width: 100%; }}
    .chat-msg-user {{
        background: linear-gradient(135deg, #b6fcd5 0%, #b5fff9 100%);
        color: #225c4b;
        border-radius: 18px 18px 5px 18px;
        padding: 16px 22px;
        margin: 12px 0 12px auto;
        max-width: 70%;
        float: right;
        clear: both;
        text-align: left;
        font-size: 1.08rem;
        box-shadow: 0px 4px 18px rgba(166, 227, 233, 0.10);
        border: 1px solid #b5fff9;
        word-break: break-word;
        transition: box-shadow 0.2s;
    }}
    .chat-msg-bot {{
        background: linear-gradient(135deg, #e0f7fa 0%, #caf7e3 100%);
        color: #225c4b;
        border-radius: 18px 18px 18px 5px;
        padding: 16px 22px;
        margin: 12px auto 12px 0;
        max-width: 70%;
        float: left;
        clear: both;
        text-align: left;
        font-size: 1.08rem;
        box-shadow: 0px 4px 18px rgba(166, 227, 233, 0.08);
        border: 1px solid #caf7e3;
        word-break: break-word;
        transition: box-shadow 0.2s;
    }}
    .chat-ref {{
        color: #3bb273;
        font-size: 0.89em;
        margin-top: 10px;
        padding-left: 5px;
        display: block;
        text-align: left;
        clear: both;
        font-style: italic;
    }}
    .stChatMessage > div {{ overflow: auto; }}
    .chat-header-sticky {{
        position: sticky;
        top: 0;
        z-index: 99;
        background: linear-gradient(90deg, #b5fff9 0%, #a6e3e9 100%);
        color: #225c4b;
        padding: 1.2rem 1rem 1.2rem 1rem;
        border-radius: 0 0 16px 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(166,227,233,0.09);
        text-align: center;
        font-size: 2.1rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .chat-input-bar input {{
        border-radius: 20px;
        border: 1.5px solid #caf7e3;
        padding: 12px 18px;
        font-size: 1.08rem;
        background: #e3fdfd;
        outline: none;
        transition: border 0.2s;
    }}
    .chat-input-bar input:focus {{
        border: 1.5px solid #a6e3e9;
        background: #b6fcd5;
    }}
    .stButton > button {{
        background: linear-gradient(90deg, #b5fff9 0%, #b6fcd5 100%);
        color: #225c4b;
        border: 0;
        border-radius: 20px;
        font-size: 1.08rem;
        font-weight: 600;
        padding: 10px 32px;
        margin-top: 8px;
        box-shadow: 0 2px 8px rgba(166,227,233,0.08);
        transition: background 0.2s, color 0.2s;
    }}
    .stButton > button:hover {{
        background: linear-gradient(90deg, #b6fcd5 0%, #b5fff9 100%);
        color: #176d5a;
    }}
    /* Scrollbar bonito */
    ::-webkit-scrollbar {{ width: 10px; background: #caf7e3; border-radius: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: #b6fcd5; border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)

# --- Global Variable for PDF Chunks ---
PDF_CHUNKS_GLOBAL = []

# --- Functions PDF / Context Retrieval / LLM ---

# @st.cache_data
def load_pdf_and_chunks(pdf_path):
    global PDF_CHUNKS_GLOBAL
    PDF_CHUNKS_GLOBAL = []
    if not os.path.exists(pdf_path): st.error(f"Error Crítico: PDF no encontrado: '{os.path.abspath(pdf_path)}'"); return None
    # --- !!! CORRECCIÓN SyntaxError AQUÍ !!! ---
    chunks = [] # Inicialización en su línea
    try:        # try: en la siguiente línea
    # -----------------------------------------
        print(f"Leyendo PDF: {pdf_path}")
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            print(f"{num_pages} páginas.")
            if num_pages == 0: st.warning("PDF sin páginas.")
            for i, page in enumerate(reader.pages):
                try:
                    pg_text = page.extract_text() or ""
                    if not pg_text.strip(): continue
                    paragraphs = re.split(r'\n\s*\n+', pg_text)
                    for par_num, par in enumerate(paragraphs):
                         texto_limpio_par = re.sub(r'\s+', ' ', par).strip()
                         if len(texto_limpio_par) > 40: chunks.append({'pagina': i + 1, 'texto': texto_limpio_par})
                except Exception as page_e:
                    print(f"Error pág {i+1}: {page_e}")
                    st.warning(f"Error procesando pág {i+1}.")
        print(f"{len(chunks)} chunks encontrados.")
        if not chunks: st.warning("Advertencia: No se extrajeron chunks válidos.")
        PDF_CHUNKS_GLOBAL = chunks
        return chunks
    except Exception as e:
        st.error(f"Error Crítico PDF: {e}")
        print(traceback.format_exc())
        return None

def find_relevant_context(pregunta, chunks, top_n=3):
    # (Sin cambios en esta función)
    if not chunks or not pregunta or not pregunta.strip(): return None, []
    try:
        corpus = [c['texto'] for c in chunks];
        if not corpus: return None, []
        vectorizer = TfidfVectorizer(); corpus_vecs = vectorizer.fit_transform(corpus)
        pregunta_vec = vectorizer.transform([pregunta]); similitudes = cosine_similarity(pregunta_vec, corpus_vecs)[0]
        relevant_indices = [i for i, score in sorted(enumerate(similitudes), key=lambda item: item[1], reverse=True) if score > 0.01]
        if not relevant_indices: return None, []
        top_indices = relevant_indices[:min(top_n, len(relevant_indices))]
        contexto_combinado = ""; paginas_fuente = set()
        for idx in top_indices:
            chunk_data = chunks[idx]; contexto_combinado += f"---\nPág ~ {chunk_data['pagina']}:\n{chunk_data['texto']}\n\n"; paginas_fuente.add(chunk_data['pagina'])
        return contexto_combinado.strip(), sorted(list(paginas_fuente))
    except Exception as e: st.error(f"Error búsqueda TF-IDF: {e}"); print(traceback.format_exc()); return None, []

def get_llm_response(system_prompt, user_prompt, model_name, api_url):
    # (Sin cambios en esta función)
    headers = {'Content-Type': 'application/json'}
    payload = { "model": model_name, "messages": [{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}], "temperature": 0.5, "max_tokens": 1000, "stream": False }
    print(f"DEBUG: Conectando a: {api_url}, Modelo API: {model_name}")
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=180)
        print(f"LLM Status: {response.status_code}")
        if response.status_code == 404: st.error(f"Error 404: Endpoint no encontrado: {api_url}"); return f"Error: Endpoint {api_url} no encontrado."
        if response.status_code == 400: st.error(f"Error 400: Petición incorrecta. ¿Modelo '{model_name}' válido? Detalle: {response.text}"); return f"Error: Petición incorrecta (¿modelo?)."
        response.raise_for_status()
        try:
            response_data = response.json(); print("Respuesta LLM recibida.")
            # print(f"Datos crudos: {response_data}")
            if 'choices' in response_data and isinstance(response_data['choices'], list) and response_data['choices']:
                if 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
                     print("DEBUG: Extracción OpenAI exitosa."); return response_data['choices'][0]['message']['content'].strip()
            keys = list(response_data.keys()) if isinstance(response_data, dict) else "N/A"; print(f"Advertencia: Formato OpenAI inesperado. Claves: {keys}")
            st.warning("Formato respuesta LLM inesperado."); return f"Error: Formato OpenAI no reconocido. Respuesta:\n```json\n{json.dumps(response_data, indent=2)}\n```"
        except json.JSONDecodeError: st.error(f"Error: Respuesta LLM no es JSON válido."); return f"Error: Respuesta inválida LLM: {response.text}"
    except requests.exceptions.Timeout: st.error("Error: Timeout LLM."); return "Error: Timeout."
    except requests.exceptions.RequestException as e: st.error(f"Error Conexión: {e}"); return f"Error Conexión: {e}"
    except Exception as e: print(traceback.format_exc()); st.error(f"Error inesperado LLM: {e}"); return f"Error inesperado: {e}"

# --- === App Execution Starts Here === --

st.markdown(f"<div class='chat-header-sticky'>💼 Chatbot RRHH ({DISPLAY_MODEL_NAME})</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: #546e7a; font-size:1.12rem;'>Consulte información del manual del empleado y el Código del Trabajo-Chile usando IA local.</p>", unsafe_allow_html=True)

# --- Cargar rutas de documentos ---
pdf_file_path = r"C:\Users\myazi\ChatBot\Prueba4.pdf"  # Manual del empleado
codigo_trabajo_path = r"C:\Users\myazi\ChatBot\Código del Trabajo-Chile.pdf"  # Código del Trabajo-Chile

# --- Cargar ambos documentos y combinar chunks ---
manual_chunks = load_pdf_and_chunks(pdf_file_path) or []
codigo_trabajo_chunks = load_pdf_and_chunks(codigo_trabajo_path) or []
# Añadir un campo 'fuente' a cada chunk para identificar el origen
def add_source(chunks, fuente):
    for c in chunks:
        c['fuente'] = fuente
add_source(manual_chunks, os.path.basename(pdf_file_path))
add_source(codigo_trabajo_chunks, os.path.basename(codigo_trabajo_path))
# Unir todos los chunks
todos_los_chunks = manual_chunks + codigo_trabajo_chunks

# --- Initialize Chat State ---
if "chat" not in st.session_state:
    st.session_state["chat"] = []
    if todos_los_chunks:
        st.session_state["chat"].append({'role': 'bot', 'msg': '¡Hola! Soy tu asistente virtual de RRHH. ¿En qué puedo ayudarte respecto al manual del empleado o el Código del Trabajo?', 'ref': 'Inicial'})
    else:
        st.session_state["chat"].append({'role': 'bot', 'msg': f'⚠️ **Error Crítico:** No se pudo cargar uno o ambos documentos.', 'ref': 'Error Carga'})

# --- Processing Function (Callback) ---
def process_user_input():
    user_question = st.session_state.get("chat_input_key", None)
    if not todos_los_chunks:
        st.error("Procesamiento detenido: Documentos no cargados.")
        return
    if not user_question or not user_question.strip():
        return

    user_question_clean = user_question.strip()
    print(f"\n--- Procesando Consulta ---\nUsuario: {user_question_clean}")
    st.session_state.chat.append({'role': 'user', 'msg': user_question_clean})
    status = st.status("Procesando...", expanded=False)

    try:
        status.update(label="Analizando documentos...")
        contexto, paginas_fuente = find_relevant_context(user_question_clean, todos_los_chunks, top_n=3)

        # --- Limitar el contexto para evitar error de tokens ---
        MAX_CONTEXT_CHARS = 120000  # Aproximadamente 32768 tokens
        contexto_recortado = contexto
        contexto_truncado = False
        if contexto and len(contexto) > MAX_CONTEXT_CHARS:
            contexto_recortado = contexto[:MAX_CONTEXT_CHARS]
            contexto_truncado = True

        prompt_contexto = "Contexto de los documentos:\n{contexto}\n\n---\nPregunta del Usuario: {pregunta}\n---\nInstrucciones: Cita documento y páginas si es posible. Si no está en contexto, indícalo. Sé profesional y conciso."
        prompt_sin_contexto = "Pregunta del Usuario: {pregunta}\n---\nInstrucciones: No hay contexto de los documentos. Si no estás seguro, indícalo. No inventes. Sé profesional y conciso."
        system_prompt = f"Eres un asistente experto de RRHH para la Asociación Pro Desarrollo Comunal del Patio, Inc. Cultura FORMAL E INNOVADORA. Responde basándote *estrictamente* en el contexto de los documentos (Manual: {os.path.basename(pdf_file_path)}, Código: {os.path.basename(codigo_trabajo_path)}) si se proporciona."

        user_prompt_final = ""; ref_info_for_state = []
        if contexto_recortado:
            print(f"DEBUG: Contexto encontrado. Páginas: {paginas_fuente}")
            user_prompt_final = prompt_contexto.format(contexto=contexto_recortado, pregunta=user_question_clean)
            ref_info_for_state = paginas_fuente
            label_text = f"Consultando a {DISPLAY_MODEL_NAME}..."
            if isinstance(paginas_fuente, list):
                try:
                    pages_str = ", ".join(map(str, paginas_fuente))
                    label_text = f"Contexto (Págs: {pages_str}). Consultando..."
                except Exception:
                    pass
            status.update(label=label_text)
            if contexto_truncado:
                st.info("El contexto fue recortado para evitar superar el límite del modelo. Si necesitas más detalle, haz una pregunta más específica.")
        else:
            print("Advertencia: No se encontró contexto relevante.")
            user_prompt_final = prompt_sin_contexto.format(pregunta=user_question_clean)
            ref_info_for_state = ["Contexto no encontrado"]
            status.update(label=f"Consultando a {DISPLAY_MODEL_NAME}...")

        llm_response = get_llm_response(system_prompt, user_prompt_final, LLM_MODEL_NAME, LLM_API_URL)
        st.session_state.chat.append({'role': 'bot', 'msg': llm_response, 'ref': ref_info_for_state})
        status.update(label="Respuesta lista.", state="complete", expanded=False)

    except Exception as e:
        print(f"Error en process_user_input: {e}"); print(traceback.format_exc())
        st.error(f"Ocurrió un error al procesar tu pregunta: {e}")
        error_message = f"Error al procesar: {type(e).__name__}."
        st.session_state.chat.append({'role': 'bot', 'msg': error_message, 'ref': 'Error Proceso'})
        status.update(label="Error", state="error", expanded=True)

# --- Display Existing Chat Messages ---
st.divider()
chat_placeholder = st.container()
with chat_placeholder:
    if "chat" in st.session_state:
        for i, turno in enumerate(st.session_state.chat):
            try:
                is_user = turno['role'] == 'user'
                avatar_icon = "🧑‍💻" if is_user else "💼"
                with st.chat_message(turno['role'], avatar=avatar_icon):
                    css_class = "chat-msg-user" if is_user else "chat-msg-bot"
                    msg_content = turno.get('msg', '*Mensaje no disponible*')
                    st.markdown(f'<div class="{css_class}">{msg_content}</div>', unsafe_allow_html=True)
                    if not is_user:
                        ref_info = turno.get('ref')
                        if ref_info and ref_info not in ['Inicial', 'Error Carga', 'Error Proceso', ["Contexto no encontrado"], ["N/A"]]:
                            try:
                                # Mostrar referencias de ambos documentos
                                if isinstance(ref_info, list) and ref_info and isinstance(ref_info[0], dict):
                                    for ref in ref_info:
                                        st.markdown(f'<div class="chat-ref">Referencia: {ref.get("fuente", "¿?")}, pág. ~ {ref.get("pagina", "¿?")}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="chat-ref">Referencia: {ref_info}</div>', unsafe_allow_html=True)
                            except Exception as ref_e:
                                print(f"Error display ref: {ref_e}"); st.caption(f"Ref: {ref_info} (Error)")
                        elif ref_info == ["Contexto no encontrado"]:
                             st.markdown(f'<div class="chat-ref">Nota: No se encontró contexto específico en los documentos.</div>', unsafe_allow_html=True)
            except Exception as display_e:
                print(f"Error display msg {i}: {display_e}"); st.error(f"Error al mostrar mensaje {i}."); print(traceback.format_exc())

# --- Chat Input Widget ---
st.chat_input(
    f"Pregunta a {DISPLAY_MODEL_NAME} sobre el manual o el Código del Trabajo...",
    key="chat_input_key",
    on_submit=process_user_input
)

# --- Footer ---
st.caption(f"Asistente RRHH v1.9.2 | Potenciado por: {DISPLAY_MODEL_NAME} | Documentos: {os.path.basename(pdf_file_path)}, {os.path.basename(codigo_trabajo_path)}")