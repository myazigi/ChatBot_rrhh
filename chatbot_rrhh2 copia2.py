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

# --- Título Moderno y Atractivo ---
st.markdown("""
<div style="
    text-align: center;
    font-size: 2.7rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #b85c38;
    margin-top: 0.5em;
    margin-bottom: 1.2em;
    padding: 0.6em 0 0.6em 0;
    background: rgba(247, 178, 103, 0.18);
    border-radius: 22px;
    box-shadow: 0 2px 12px rgba(184,92,56,0.07);
    backdrop-filter: blur(2px);
">
    💼 Chatbot RRHH IA
</div>
""", unsafe_allow_html=True)

# --- LLM Backend Configuration ---
LLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_MODEL_NAME = "gemma-3-1b-it-qat" # ¡VERIFICA ESTO! 

# --- Crear Nombre Corto para Mostrar ---
try:
    name_parts = re.split(r'[:/-]', LLM_MODEL_NAME)
    DISPLAY_MODEL_NAME = name_parts[0].capitalize() if name_parts else LLM_MODEL_NAME
    if not DISPLAY_MODEL_NAME: DISPLAY_MODEL_NAME = "LLM Local"
except Exception: DISPLAY_MODEL_NAME = "LLM Local"

# --- CSS Modernizado y Glassmorphism ---
st.markdown("""
<style>
    /* Fondo principal */
    section.main {
        background: linear-gradient(120deg, #f9dbbd 0%, #ffe5d9 80%, #e7c6a9 100%) !important;
    }
    /* Contenedor principal de la app */
    .block-container {
        background: rgba(255, 247, 240, 0.55) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px 0 rgba(184, 92, 56, 0.10) !important;
        padding: 2.5rem !important;
        backdrop-filter: blur(10px) !important;
        border: 1.5px solid rgba(231, 198, 169, 0.18) !important;
    }
    /* Botón principal */
    .stButton > button {
        background: linear-gradient(90deg, #f7b267 0%, #e7c6a9 100%) !important;
        color: #b85c38 !important;
        border-radius: 18px !important;
        font-size: 1.13rem !important;
        font-weight: 600 !important;
        padding: 12px 38px !important;
        margin-top: 10px !important;
        box-shadow: 0 4px 18px rgba(184,92,56,0.09) !important;
        letter-spacing: 0.02em !important;
        border: none !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #e7c6a9 0%, #f7b267 100%) !important;
        color: #6b3e26 !important;
        transform: translateY(-2px) scale(1.04) !important;
    }
    /* Input de texto */
    .stTextInput input {
        border-radius: 16px !important;
        border: 1.5px solid #e7c6a9 !important;
        padding: 14px 20px !important;
        font-size: 1.13rem !important;
        background: rgba(255, 247, 240, 0.85) !important;
        outline: none !important;
        transition: border 0.2s !important;
    }
    .stTextInput input:focus {
        border: 1.5px solid #f4845f !important;
        background: #fff7f0 !important;
    }
    /* Chat burbujas */
    .chat-msg-user {
        background: rgba(247, 178, 103, 0.75);
        color: #6b3e26;
        border-radius: 12px 12px 4px 12px;
        padding: 18px 28px;
        margin: 14px 0 14px auto;
        max-width: 65%;
        float: right;
        clear: both;
        text-align: left;
        font-size: 1.15rem;
        font-weight: 500;
        box-shadow: 0 6px 22px 0 rgba(184, 92, 56, 0.08);
        border: 1.5px solid rgba(249, 219, 189, 0.45);
        word-break: break-word;
        transition: box-shadow 0.2s;
        backdrop-filter: blur(2px);
    }
    .chat-msg-bot {
        background: rgba(255, 229, 217, 0.75);
        color: #7d4f2a;
        border-radius: 12px 12px 12px 4px;
        padding: 18px 28px;
        margin: 14px auto 14px 0;
        max-width: 65%;
        float: left;
        clear: both;
        text-align: left;
        font-size: 1.15rem;
        font-weight: 500;
        box-shadow: 0 6px 22px 0 rgba(231, 198, 169, 0.10);
        border: 1.5px solid rgba(249, 219, 189, 0.35);
        word-break: break-word;
        transition: box-shadow 0.2s;
        backdrop-filter: blur(2px);
    }
    .chat-ref {
        color: #b85c38;
        font-size: 0.93em;
        margin-top: 10px;
        padding-left: 5px;
        display: block;
        text-align: left;
        clear: both;
        font-style: italic;
        opacity: 0.85;
    }
    /* Scrollbar personalizado */
    ::-webkit-scrollbar { width: 10px; background: #ffe5d9; border-radius: 8px; }
    ::-webkit-scrollbar-thumb { background: #f7b267; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- Global Variable for PDF Chunks ---
VECTORIZE_GLOBAL = None
CORPUS_VECS_GLOBAL = None
CHUNKS_GLOBAL = None

# --- Functions PDF / Context Retrieval / LLM ---

# --- Cargar rutas de documentos ---
pdf_file_path = r"C:\Users\myazi\ChatBot\Prueba4.pdf"  # Manual del empleado
codigo_trabajo_path = r"C:\Users\myazi\ChatBot\Código del Trabajo-Chile.pdf"  # Código del Trabajo-Chile

# --- Cargar ambos documentos y combinar chunks pequeños (~200 caracteres) ---
@st.cache_data(show_spinner="Cargando y procesando documentos PDF...")
def load_pdf_and_chunks_small(path):
    try:
        import PyPDF2
        import re
        reader = PyPDF2.PdfReader(path)
        chunks = []
        for i, page in enumerate(reader.pages):
            pg_text = page.extract_text() or ""
            if not pg_text.strip(): continue
            # Divide en bloques pequeños
            pg_text = re.sub(r'\s+', ' ', pg_text)
            for start in range(0, len(pg_text), 200):
                chunk = pg_text[start:start+200].strip()
                if len(chunk) > 40:
                    chunks.append({'pagina': i + 1, 'texto': chunk})
        return chunks
    except Exception as e:
        st.error(f"Error Crítico PDF: {e}")
        print(traceback.format_exc())
        return None

@st.cache_data(show_spinner="Calculando embeddings de contexto...")
def precalc_tfidf(chunks):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    if not chunks:
        return None, None, None
    corpus = [c['texto'] for c in chunks]
    vectorizer = TfidfVectorizer()
    corpus_vecs = vectorizer.fit_transform(corpus)
    return corpus, vectorizer, corpus_vecs

# --- Cargar y cachear todo al inicio ---
manual_chunks = load_pdf_and_chunks_small(pdf_file_path) or []
codigo_trabajo_chunks = load_pdf_and_chunks_small(codigo_trabajo_path) or []
todos_los_chunks = manual_chunks + codigo_trabajo_chunks
CORPUS, VECTORIZE_GLOBAL, CORPUS_VECS_GLOBAL = precalc_tfidf(todos_los_chunks)
CHUNKS_GLOBAL = todos_los_chunks

# --- Función rápida de contexto ---
def find_relevant_context(pregunta, chunks=None, top_n=1):
    if chunks is None:
        chunks = CHUNKS_GLOBAL
    if not chunks or not pregunta or not pregunta.strip() or VECTORIZE_GLOBAL is None or CORPUS_VECS_GLOBAL is None:
        return None, []
    try:
        pregunta_vec = VECTORIZE_GLOBAL.transform([pregunta])
        from sklearn.metrics.pairwise import cosine_similarity
        similitudes = cosine_similarity(pregunta_vec, CORPUS_VECS_GLOBAL)[0]
        relevant_indices = [i for i, score in sorted(enumerate(similitudes), key=lambda item: item[1], reverse=True) if score > 0.01]
        if not relevant_indices: return None, []
        top_indices = relevant_indices[:min(top_n, len(relevant_indices))]
        contexto_combinado = ""; paginas_fuente = set()
        for idx in top_indices:
            chunk_data = chunks[idx]; contexto_combinado += f"---\nPág ~ {chunk_data['pagina']}:\n{chunk_data['texto']}\n\n"; paginas_fuente.add(chunk_data['pagina'])
        return contexto_combinado.strip(), sorted(list(paginas_fuente))
    except Exception as e:
        st.error(f"Error búsqueda TF-IDF: {e}")
        print(traceback.format_exc())
        return None, []

# --- Feedback inmediato al usuario ---
def process_user_input():
    import re, traceback
    user_question = st.session_state.get('chat_input_key', '')
    if not user_question.strip(): return
    user_question_clean = user_question.strip()
    st.session_state.chat.append({'role': 'user', 'msg': user_question_clean, 'ref': None})
    status = st.status(label="Procesando...", expanded=True)

    # --- ASEGURAR PROMPTS DEFINIDOS ---
    prompt_con_contexto = """Contexto de los documentos:\n{contexto}\n\n---\nPregunta del Usuario: {pregunta}\n---\nInstrucciones: Responde de manera ejecutiva, precisa y acertiva. Explica brevemente el fundamento legal citado y su aplicación práctica. No repitas solo el texto legal: interpreta y resume para que el usuario entienda claramente la respuesta. Si corresponde, cita documento y páginas. Si no está en contexto, indícalo."""
    prompt_sin_contexto = """Pregunta del Usuario: {pregunta}\n---\nInstrucciones: No hay contexto de los documentos. Responde de manera ejecutiva, precisa y acertiva. Explica brevemente el fundamento legal citado y su aplicación práctica. No repitas solo el texto legal: interpreta y resume para que el usuario entienda claramente la respuesta. Si no estás seguro, indícalo. No inventes."""
    system_prompt = f"Eres un asistente experto de RRHH para la Asociación Pro Desarrollo Comunal del Patio, Inc. Cultura FORMAL E INNOVADORA. Responde basándote *estrictamente* en el contexto de los documentos (Manual: {os.path.basename(pdf_file_path)}, Código: {os.path.basename(codigo_trabajo_path)}) si se proporciona."

    try:
        with st.spinner("Pensando y buscando contexto..."):
            contexto, paginas_fuente = find_relevant_context(user_question_clean, top_n=2)
        if contexto:
            ref_info_for_state = paginas_fuente
            label_text = f"Consultando a {DISPLAY_MODEL_NAME}..."
            if isinstance(paginas_fuente, list):
                try:
                    pages_str = ", ".join(map(str, paginas_fuente))
                    label_text = f"Contexto (Págs: {pages_str}). Consultando..."
                except Exception:
                    pass
            status.update(label=label_text)
            if len(contexto) > 1800:
                contexto = contexto[:1800]  # Limita el contexto por si acaso
                st.info("El contexto fue recortado para evitar superar el límite del modelo. Si necesitas más detalle, haz una pregunta más específica.")
        else:
            print("Advertencia: No se encontró contexto relevante.")
            user_prompt_final = prompt_sin_contexto.format(pregunta=user_question_clean)
            ref_info_for_state = ["Contexto no encontrado"]
            status.update(label=f"Consultando a {DISPLAY_MODEL_NAME}...")
        # --- Llama al LLM ---
        with st.spinner("Consultando modelo IA..."):
            llm_response = get_llm_response(system_prompt, user_prompt_final if not contexto else prompt_con_contexto.format(contexto=contexto, pregunta=user_question_clean), LLM_MODEL_NAME, LLM_API_URL)
        # --- FILTRAR SOLO RESPUESTA ---
        import re
        matches = list(re.finditer(r"(?:Respuesta(?: final)?\s*:?|Respuesta directa\s*:?)\s*(.+)", llm_response, re.IGNORECASE|re.DOTALL))
        if matches:
            respuesta_final = matches[-1].group(1).strip()
        else:
            razonamiento_regex = r"(?:Pensamiento|Razonamiento|Explicaci[oó]n|Motivaci[oó]n|Justificaci[oó]n|An[aá]lisis)\s*:?(.+?)(?:\n|$)"
            respuesta = llm_response
            respuesta = re.sub(razonamiento_regex, '', respuesta, flags=re.IGNORECASE|re.DOTALL)
            respuesta_final = respuesta.strip()  # Mostrar respuesta completa, no solo la primera línea
            if not respuesta_final or respuesta_final.lower() in ["rrhhbot", "bot", "respuesta", "", "."] or len(respuesta_final) < 5:
                respuesta_final = llm_response.strip()
        st.session_state.chat.append({'role': 'bot', 'msg': respuesta_final, 'ref': ref_info_for_state})
        status.update(label="Respuesta lista.", state="complete", expanded=False)
    except Exception as e:
        print(f"Error en process_user_input: {e}"); print(traceback.format_exc())
        st.error(f"Ocurrió un error al procesar tu pregunta: {e}")
        error_message = f"Error al procesar: {type(e).__name__}."
        st.session_state.chat.append({'role': 'bot', 'msg': error_message, 'ref': 'Error Proceso'})
        status.update(label="Error", state="error", expanded=True)

# --- get_llm_response ---
def get_llm_response(system_prompt, user_prompt, model_name, api_url):
    headers = {'Content-Type': 'application/json'}
    payload = { "model": model_name, "messages": [{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}], "temperature": 0.5, "max_tokens": 4096, "stream": False }
    print(f"DEBUG: Conectando a: {api_url}, Modelo API: {model_name}")
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=120)
        print(f"LLM Status: {response.status_code}")
        if response.status_code == 404: st.error(f"Error 404: Endpoint no encontrado: {api_url}"); return f"Error: Endpoint {api_url} no encontrado."
        if response.status_code == 400: st.error(f"Error 400: Petición incorrecta. ¿Modelo '{model_name}' válido? Detalle: {response.text}"); return f"Error: Petición incorrecta (¿modelo?)."
        response.raise_for_status()
        try:
            response_data = response.json(); print("Respuesta LLM recibida.")
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

# --- Initialize Chat State ---
if "chat" not in st.session_state:
    st.session_state["chat"] = []
    if todos_los_chunks:
        st.session_state["chat"].append({'role': 'bot', 'msg': '¡Hola! Soy tu asistente virtual de RRHH. ¿En qué puedo ayudarte respecto al manual del empleado o el Código del Trabajo?', 'ref': 'Inicial'})
    else:
        st.session_state["chat"].append({'role': 'bot', 'msg': f'⚠️ **Error Crítico:** No se pudo cargar uno o ambos documentos.', 'ref': 'Error Carga'})

# --- Display Existing Chat Messages ---
st.divider()
chat_placeholder = st.container()
with chat_placeholder:
    if "chat" in st.session_state:
        for i, turno in enumerate(st.session_state["chat"]):
            try:
                is_user = turno['role'] == 'user'
                css_class = "chat-msg-user" if is_user else "chat-msg-bot"
                avatar = "🧑‍💻" if is_user else "💼"
                # Renderiza el mensaje con markdown y HTML personalizado
                st.markdown(f'<div class="{css_class}"><b>{avatar} {"Tú" if is_user else "RRHHBot"}</b><br>{turno.get("msg", "*Mensaje no disponible*")}</div>', unsafe_allow_html=True)
                # Referencias/contexto para respuestas del bot
                if not is_user:
                    ref_info = turno.get('ref')
                    if ref_info and ref_info not in ['Inicial', 'Error Carga', 'Error Proceso', ["Contexto no encontrado"], ["N/A"]]:
                        try:
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