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
import tempfile

# --- Streamlit config: ¡Debe ser lo primero! ---
st.set_page_config(page_title="Chatbot RRHH IA", page_icon="💼", layout="wide")

# --- LLM Backend Configuration ---
LLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL_NAME = "gemma-3-1b-it" # ¡VERIFICA ESTO!

# --- Crear Nombre Corto para Mostrar ---
try:
    name_parts = re.split(r'[:/-]', LLM_MODEL_NAME)
    DISPLAY_MODEL_NAME = name_parts[0].capitalize() if name_parts else LLM_MODEL_NAME
    if not DISPLAY_MODEL_NAME: DISPLAY_MODEL_NAME = "LLM Local"
except Exception: DISPLAY_MODEL_NAME = "LLM Local"

# --- CSS Modernizado ---
st.markdown(f"""
<style>
    .main .block-container {{ max-width: 950px; padding-top: 2rem; padding-bottom: 2rem; margin: auto; }}
    .stChatMessage {{ width: 100%; }}
    .chat-msg-user {{ background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%); color: #004d40; border-radius: 20px 20px 5px 20px; padding: 14px 18px; margin: 6px 0 6px auto; max-width: 75%; float: right; clear: both; text-align: left; font-size: 1.0rem; box-shadow: 0px 4px 10px rgba(0, 150, 136, 0.15); word-wrap: break-word; border: 1px solid #b2dfdb; }}
    .chat-msg-bot {{ background: #ffffff; color: #37474f; border-radius: 20px 20px 20px 5px; padding: 14px 18px; margin: 6px auto 6px 0; max-width: 75%; float: left; clear: both; text-align: left; font-size: 1.0rem; box-shadow: 0px 4px 10px rgba(55, 71, 79, 0.1); word-wrap: break-word; border: 1px solid #eceff1; }}
    .chat-ref {{ color: #78909c; font-size: 0.85em; margin-top: 10px; padding-left: 5px; display: block; text-align: left; clear: both; font-style: italic; }}
    .stChatMessage > div {{ overflow: auto; }}
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

# --- === App Execution Starts Here === ---

st.markdown(f"<h1 style='text-align:center; color: #263238;'>💼 Chatbot RRHH ({DISPLAY_MODEL_NAME})</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: #546e7a;'>Consulte información del manual del empleado usando IA local.</p>", unsafe_allow_html=True)

# --- PDF File Uploader ---
pdf_file = st.file_uploader("Sube el manual del empleado en PDF", type=["pdf"])

if pdf_file is not None:
    # Guardar PDF subido en memoria temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        pdf_file_path = tmp_file.name
    loaded_chunks = load_pdf_and_chunks(pdf_file_path)
else:
    pdf_file_path = None
    loaded_chunks = None

# --- Initialize Chat State ---
if "chat" not in st.session_state:
    st.session_state["chat"] = []
    if loaded_chunks is not None:
        st.session_state["chat"].append({'role': 'bot', 'msg': '¡Hola! Soy tu asistente virtual de RRHH. ¿En qué puedo ayudarte respecto al manual del empleado?', 'ref': 'Inicial'})
    else:
        st.session_state["chat"].append({'role': 'bot', 'msg': 'Por favor sube un manual de empleado en PDF para comenzar.', 'ref': 'Error Carga'})

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
                                ref_text = ", ".join(map(str, ref_info)) if isinstance(ref_info, list) else str(ref_info)
                                st.markdown(f'<div class="chat-ref">Referencia: {os.path.basename(pdf_file_path) if pdf_file_path else "PDF no cargado"}, pág(s). ~ {ref_text}</div>', unsafe_allow_html=True)
                            except Exception as ref_e: print(f"Error display ref: {ref_e}"); st.caption(f"Ref: {ref_info} (Error)")
                        elif ref_info == ["Contexto no encontrado"]:
                             st.markdown(f'<div class="chat-ref">Nota: No se encontró contexto específico en el documento.</div>', unsafe_allow_html=True)
            except Exception as display_e: print(f"Error display msg {i}: {display_e}"); st.error(f"Error al mostrar mensaje {i}."); print(traceback.format_exc())


# --- Processing Function (Callback) ---
def process_user_input():
    user_question = st.session_state.get("chat_input_key", None)
    if loaded_chunks is None:
        st.error("Procesamiento detenido: PDF no cargado.")
        return
    if not user_question or not user_question.strip():
        return

    user_question_clean = user_question.strip()
    print(f"\n--- Procesando Consulta ---\nUsuario: {user_question_clean}")
    st.session_state.chat.append({'role': 'user', 'msg': user_question_clean})
    status = st.status("Procesando...", expanded=False)

    try:
        status.update(label="Analizando documento...")
        contexto, paginas_fuente = find_relevant_context(user_question_clean, loaded_chunks, top_n=3)

        prompt_contexto = "Contexto del Manual:\n{contexto}\n\n---\nPregunta del Usuario: {pregunta}\n---\nInstrucciones: Cita páginas si es posible. Si no está en contexto, indícalo. Sé profesional y conciso."
        prompt_sin_contexto = "Pregunta del Usuario: {pregunta}\n---\nInstrucciones: No hay contexto del manual. Si no estás seguro, indícalo. No inventes. Sé profesional y conciso."
        system_prompt = f"Eres un asistente experto de RRHH para la Asociación Pro Desarrollo Comunal del Patio, Inc. Cultura FORMAL E INNOVADORA. Responde basándote *estrictamente* en el contexto del manual ({os.path.basename(pdf_file_path) if pdf_file_path else 'PDF no cargado'}) si se proporciona."

        user_prompt_final = ""; ref_info_for_state = []
        if contexto:
            print(f"DEBUG: Contexto encontrado. Páginas: {paginas_fuente}")
            user_prompt_final = prompt_contexto.format(contexto=contexto, pregunta=user_question_clean)
            ref_info_for_state = paginas_fuente
            label_text = f"Consultando a {DISPLAY_MODEL_NAME}..."
            if isinstance(paginas_fuente, list):
                try:
                    pages_str = ", ".join(map(str, paginas_fuente))
                    label_text = f"Contexto (Págs: {pages_str}). Consultando..."
                except Exception:
                    pass
            status.update(label=label_text)
        else:
            print("Advertencia: No se encontró contexto relevante.")
            user_prompt_final = prompt_sin_contexto.format(pregunta=user_question_clean)
            ref_info_for_state = ["Contexto no encontrado"]
            status.update(label=f"Consultando a {DISPLAY_MODEL_NAME}...")

        llm_response = get_llm_response(system_prompt, user_prompt_final, LLM_MODEL_NAME, LLM_API_URL)
        st.session_state.chat.append({'role': 'bot', 'msg': llm_response, 'ref': ref_info_for_state})
        status.update(label="Respuesta lista.", state="complete", expanded=False)

    except Exception as e:
        print(f"Error en process_user_input: {e}")
        print(traceback.format_exc())
        st.error(f"Ocurrió un error al procesar tu pregunta: {e}")
        error_message = f"Error al procesar: {type(e).__name__}."
        st.session_state.chat.append({'role': 'bot', 'msg': error_message, 'ref': 'Error Proceso'})
        status.update(label="Error", state="error", expanded=True)


# --- Chat Input Widget ---
st.chat_input(
    f"Pregunta a {DISPLAY_MODEL_NAME} sobre el manual...",
    key="chat_input_key",
    on_submit=process_user_input
)

# --- Footer ---
st.caption(f"Asistente RRHH v1.9.2 | Potenciado por: {DISPLAY_MODEL_NAME} | Documento: {os.path.basename(pdf_file_path) if pdf_file_path else 'PDF no cargado'}")