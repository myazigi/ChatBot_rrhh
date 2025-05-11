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
# Intenta obtener de variables de entorno, si no, usa valores por defecto
LLM_API_URL = os.environ.get("LLM_API_URL", "http://100.101.84.15:1234/v1/chat/completions")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemma-3-1b-it-qat") # ¡VERIFICA ESTO! Puede necesitar / en vez de -

# --- Crear Nombre Corto para Mostrar ---
try:
    # Maneja nombres como 'nombre/modelo' o 'nombre:tag'
    name_parts = re.split(r'[:/-]', LLM_MODEL_NAME)
    DISPLAY_MODEL_NAME = name_parts[-1].replace('-',' ').title() if name_parts else LLM_MODEL_NAME # Usa la última parte
    if not DISPLAY_MODEL_NAME: DISPLAY_MODEL_NAME = "LLM Local"
except Exception: DISPLAY_MODEL_NAME = "LLM Local"

# --- CSS Modernizado y Glassmorphism ---
# (CSS se mantiene igual que en la pregunta)
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
    /* Botón principal (si se usara fuera del input) */
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
        margin: 14px 0 14px auto; /* Empuja a la derecha */
        max-width: 65%;
        float: right; /* Asegura alineación derecha */
        clear: both;
        text-align: left;
        font-size: 1.15rem;
        font-weight: 500;
        box-shadow: 0 6px 22px 0 rgba(184, 92, 56, 0.08);
        border: 1.5px solid rgba(249, 219, 189, 0.45);
        word-break: break-word;
        transition: box-shadow 0.2s;
        backdrop-filter: blur(2px);
        display: inline-block; /* <-- Añadido para evitar el despliegue vertical */
        white-space: pre-line; /* <-- Añadido para respetar saltos de línea y evitar corte vertical */
    }
    .chat-msg-bot {
        background: rgba(255, 229, 217, 0.75);
        color: #7d4f2a;
        border-radius: 12px 12px 12px 4px;
        padding: 18px 28px;
        margin: 14px auto 14px 0; /* Empuja a la izquierda */
        max-width: 65%;
        float: left; /* Asegura alineación izquierda */
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
    /* Referencia debajo de burbuja bot */
    .chat-ref {
        color: #b85c38;
        font-size: 0.93em;
        margin-top: -8px; /* Más cerca de la burbuja */
        margin-bottom: 10px; /* Espacio antes del siguiente mensaje */
        padding-left: 10px; /* Alineado un poco a la derecha */
        display: block;
        text-align: left;
        clear: both; /* Asegura que esté debajo */
        float: left; /* Alinea a la izquierda */
        width: 65%; /* Coincide con max-width del bot */
        font-style: italic;
        opacity: 0.85;
    }
    /* Scrollbar personalizado */
    ::-webkit-scrollbar { width: 10px; background: #ffe5d9; border-radius: 8px; }
    ::-webkit-scrollbar-thumb { background: #f7b267; border-radius: 8px; }

    /* Contenedor para el input fijo (opcional, si se desea) */
    /* .stChatInputContainer { position: sticky; bottom: 0; background: rgba(255, 247, 240, 0.8); backdrop-filter: blur(5px); padding: 1rem 0; } */
</style>
""", unsafe_allow_html=True)

# --- Global Variable Placeholders (will be populated by cached functions) ---
DOC_LOAD_ERROR = False
VECTORIZER_GLOBAL: Optional[TfidfVectorizer] = None
CORPUS_VECS_GLOBAL: Optional[Any] = None # Type depends on sparse matrix format
CHUNKS_GLOBAL: Optional[List[Dict[str, Any]]] = None
CORPUS_GLOBAL: Optional[List[str]] = None

# --- Functions PDF / Context Retrieval / LLM ---

# --- Cargar rutas de documentos ---
# Usar rutas relativas o absolutas según necesidad. Es mejor si están en el mismo dir o subdir.
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
pdf_file_path = os.path.join(script_dir, "Prueba4.pdf") # Manual del empleado
codigo_trabajo_path = os.path.join(script_dir, "Código del Trabajo-Chile.pdf") # Código del Trabajo-Chile

# --- Verificar existencia de archivos ---
if not os.path.exists(pdf_file_path):
    st.error(f"Error Crítico: No se encontró el archivo del manual en '{pdf_file_path}'")
    DOC_LOAD_ERROR = True
if not os.path.exists(codigo_trabajo_path):
    st.error(f"Error Crítico: No se encontró el archivo del Código del Trabajo en '{codigo_trabajo_path}'")
    DOC_LOAD_ERROR = True

# --- Cargar PDF y dividir en chunks ---
# Aumentamos el tamaño del chunk para mejor contexto
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50 # Pequeño solapamiento para no cortar ideas abruptamente

@st.cache_data(show_spinner="Cargando y procesando documentos PDF...")
def load_and_chunk_pdfs(manual_path: str, codigo_path: str) -> List[Dict[str, Any]]:
    """Carga ambos PDFs, los divide en chunks y añade metadatos."""
    all_chunks = []
    doc_sources = {
        "Manual": manual_path,
        "Código Trabajo": codigo_path
    }

    for doc_name, path in doc_sources.items():
        if not os.path.exists(path):
            # El error ya se mostró antes, solo saltamos este doc
            continue
        try:
            reader = PyPDF2.PdfReader(path)
            print(f"Procesando {doc_name} ({len(reader.pages)} páginas)...")
            for i, page in enumerate(reader.pages):
                pg_text = page.extract_text()
                if not pg_text or not pg_text.strip():
                    print(f"  - Página {i+1} de {doc_name} sin texto extraíble.")
                    continue

                # Limpieza básica
                pg_text = re.sub(r'\s+', ' ', pg_text).strip()
                pg_text = re.sub(r'-\n', '', pg_text) # Unir palabras cortadas por guión

                # Dividir en chunks con solapamiento
                for start in range(0, len(pg_text), CHUNK_SIZE - CHUNK_OVERLAP):
                    end = start + CHUNK_SIZE
                    chunk_text = pg_text[start:end].strip()
                    # Asegurar un tamaño mínimo razonable para el chunk
                    if len(chunk_text) > 50: # Evita chunks muy pequeños o vacíos
                        all_chunks.append({
                            'fuente': doc_name,
                            'pagina': i + 1,
                            'texto': chunk_text
                        })
            print(f"  - {doc_name} procesado.")
        except FileNotFoundError:
            # Error ya manejado al inicio
            pass
        except Exception as e:
            st.error(f"Error Crítico al procesar PDF '{doc_name}': {e}")
            print(traceback.format_exc())
            DOC_LOAD_ERROR = True

    if not all_chunks:
        st.error("Error Crítico: No se pudieron procesar chunks de ningún documento.")
        DOC_LOAD_ERROR = True

    return all_chunks

@st.cache_data(show_spinner="Calculando embeddings de contexto...")
def precalculate_tfidf_vectors(_chunks: List[Dict[str, Any]]) -> Tuple[Optional[List[str]], Optional[TfidfVectorizer], Optional[Any]]:
    """Calcula vectores TF-IDF para los chunks."""
    if not _chunks:
        return None, None, None
    try:
        corpus = [c['texto'] for c in _chunks]
        vectorizer = TfidfVectorizer(stop_words='english', # Considerar 'spanish' si aplica
                                     ngram_range=(1, 2), # Considera bigramas
                                     max_df=0.85,        # Ignora términos muy frecuentes
                                     min_df=2)           # Ignora términos muy raros
        corpus_vectors = vectorizer.fit_transform(corpus)
        print(f"TF-IDF calculado. Vocabulario: {len(vectorizer.get_feature_names_out())} términos.")
        return corpus, vectorizer, corpus_vectors
    except Exception as e:
        st.error(f"Error calculando TF-IDF: {e}")
        print(traceback.format_exc())
        return None, None, None

# --- Cargar y cachear todo al inicio (si no hay error de archivo) ---
if not DOC_LOAD_ERROR:
    CHUNKS_GLOBAL = load_and_chunk_pdfs(pdf_file_path, codigo_trabajo_path)
    if CHUNKS_GLOBAL:
        CORPUS_GLOBAL, VECTORIZER_GLOBAL, CORPUS_VECS_GLOBAL = precalculate_tfidf_vectors(CHUNKS_GLOBAL)
        if CORPUS_VECS_GLOBAL is None: # Si TF-IDF falló
             DOC_LOAD_ERROR = True
    else: # Si load_and_chunk_pdfs devolvió lista vacía
        DOC_LOAD_ERROR = True

# --- Función rápida de contexto ---
def find_relevant_context(pregunta: str, top_n: int = 3) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Encuentra los chunks más relevantes para la pregunta usando TF-IDF."""
    if DOC_LOAD_ERROR or not CHUNKS_GLOBAL or not pregunta or not pregunta.strip() or VECTORIZER_GLOBAL is None or CORPUS_VECS_GLOBAL is None:
        print("Contexto no buscado: Faltan datos o hubo error previo.")
        return None, []

    try:
        pregunta_vec = VECTORIZER_GLOBAL.transform([pregunta])
        similitudes = cosine_similarity(pregunta_vec, CORPUS_VECS_GLOBAL)[0]

        # Obtener índices y scores, ordenados por score desc
        relevant_indices_scores = sorted(enumerate(similitudes), key=lambda item: item[1], reverse=True)

        # Filtrar por un umbral mínimo de similitud (ajustable)
        threshold = 0.05 # Aumentado ligeramente el umbral
        top_indices = [idx for idx, score in relevant_indices_scores if score > threshold][:top_n]

        if not top_indices:
            print(f"No se encontró contexto relevante para: '{pregunta[:50]}...' (Umbral: {threshold})")
            return None, []

        relevant_chunks_data = []
        contexto_combinado = "Contexto Relevante de los Documentos:\n\n"
        seen_texts = set() # Para evitar duplicados exactos si hay overlap alto

        for idx in top_indices:
            chunk_data = CHUNKS_GLOBAL[idx]
            chunk_text = chunk_data['texto']

            if chunk_text not in seen_texts:
                 contexto_combinado += f"---\nFuente: {chunk_data['fuente']}, Pág. ~{chunk_data['pagina']}\n"
                 contexto_combinado += f"{chunk_text}\n\n"
                 relevant_chunks_data.append({
                     "fuente": chunk_data['fuente'],
                     "pagina": chunk_data['pagina']
                 })
                 seen_texts.add(chunk_text)

        print(f"Contexto encontrado ({len(relevant_chunks_data)} chunks) para: '{pregunta[:50]}...'")
        return contexto_combinado.strip(), relevant_chunks_data

    except Exception as e:
        st.error(f"Error durante la búsqueda de contexto TF-IDF: {e}")
        print(traceback.format_exc())
        return None, []

# --- Llamada al LLM ---
def get_llm_response(system_prompt: str, user_prompt: str, model_name: str, api_url: str) -> str:
    """Envía la solicitud al LLM local y devuelve la respuesta."""
    headers = {'Content-Type': 'application/json'}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.5, # Un poco más determinista
        "max_tokens": 8192, # Intentar permitir respuestas aún más largas
        "stream": False
    }

    print(f"DEBUG: Conectando a LLM: {api_url}, Modelo API: {model_name}")
    print(f"DEBUG: System Prompt: {system_prompt[:100]}...")
    print(f"DEBUG: User Prompt: {user_prompt[:150]}...")

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=180) # Aumentado timeout
        print(f"LLM Status Code: {response.status_code}")

        if response.status_code == 404:
            st.error(f"Error 404: Endpoint LLM no encontrado: {api_url}. Verifica la URL y si el servidor está corriendo.")
            return f"Error Crítico: Endpoint LLM ({api_url}) no encontrado."
        if response.status_code == 400:
            try:
                error_detail = response.json()
            except json.JSONDecodeError:
                error_detail = response.text
            st.error(f"Error 400: Petición incorrecta al LLM. ¿Modelo '{model_name}' válido y cargado? Detalle: {error_detail}")
            return f"Error Crítico: Petición incorrecta al LLM (¿modelo '{model_name}' correcto?)."
        
        # Lanza excepción para otros códigos de error HTTP (e.g., 500)
        response.raise_for_status()

        try:
            response_data = response.json()
            # print(f"DEBUG: Raw LLM response data: {json.dumps(response_data, indent=2)}") # Descomentar para debug profundo

            # Extraer contenido según formato OpenAI
            if 'choices' in response_data and isinstance(response_data['choices'], list) and response_data['choices']:
                choice = response_data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    content = choice['message']['content'].strip()
                    print("DEBUG: Extracción de contenido OpenAI exitosa.")
                    # A veces los modelos añaden prefijos no deseados, intentar limpiarlos
                    prefixes_to_remove = [f"{model_name}:", "respuesta:", "Respuesta:", "Bot:", "RRHHBot:"]
                    for prefix in prefixes_to_remove:
                        if content.lower().startswith(prefix.lower()):
                            content = content[len(prefix):].strip()
                    # --- DEBUG: Mostrar respuesta cruda y longitud ---
                    print("LLM RAW RESPONSE:", repr(content))
                    print("LLM RESPONSE LENGTH:", len(content))
                    # --- FIN DEBUG ---
                    return content
                elif 'text' in choice: # Algunos modelos más antiguos usan 'text'
                     content = choice['text'].strip()
                     print("DEBUG: Extracción de contenido (formato alternativo 'text') exitosa.")
                     # --- DEBUG: Mostrar respuesta cruda y longitud ---
                     print("LLM RAW RESPONSE:", repr(content))
                     print("LLM RESPONSE LENGTH:", len(content))
                     # --- FIN DEBUG ---
                     return content


            # Si no se encuentra en el formato esperado
            keys = list(response_data.keys()) if isinstance(response_data, dict) else "N/A (No es un diccionario)"
            print(f"Advertencia: Formato de respuesta LLM inesperado. Claves recibidas: {keys}")
            st.warning("Formato de respuesta del LLM inesperado. Mostrando respuesta completa.")
            # Devolver la respuesta completa como string formateado si no se pudo extraer
            return f"Respuesta LLM (formato inesperado):\n```json\n{json.dumps(response_data, indent=2)}\n```"

        except json.JSONDecodeError:
            st.error(f"Error: La respuesta del LLM no es JSON válido. Respuesta recibida:\n{response.text}")
            return f"Error Crítico: Respuesta inválida del LLM (no es JSON)."
        except KeyError as e:
             st.error(f"Error: Faltan claves esperadas en la respuesta JSON del LLM (KeyError: {e}). Respuesta: {response_data}")
             return f"Error Crítico: Formato de respuesta LLM incompleto."

    except requests.exceptions.Timeout:
        st.error("Error: Timeout esperando respuesta del LLM. El modelo podría estar tardando demasiado o el servidor no responde.")
        return "Error Crítico: Timeout del LLM."
    except requests.exceptions.ConnectionError as e:
        st.error(f"Error de Conexión: No se pudo conectar a {api_url}. ¿Está el servidor LLM corriendo y accesible? Detalles: {e}")
        return f"Error Crítico: No se pudo conectar al servidor LLM en {api_url}."
    except requests.exceptions.RequestException as e:
        st.error(f"Error en la petición al LLM: {e}")
        print(traceback.format_exc())
        return f"Error Crítico en la comunicación con el LLM: {e}"
    except Exception as e:
        st.error(f"Error inesperado durante la llamada al LLM: {e}")
        print(traceback.format_exc())
        return f"Error Crítico inesperado: {e}"


# --- Lógica Principal del Chat ---

# --- Feedback inmediato al usuario ---
def process_user_input():
    user_question = st.session_state.get('chat_input_key', '')
    if not user_question or not user_question.strip():
        st.warning("Por favor, introduce una pregunta.")
        return

    user_question_clean = user_question.strip()
    # Añadir mensaje del usuario a la historia
    st.session_state.chat.append({'role': 'user', 'msg': user_question_clean, 'ref': None})

    # Mostrar estado mientras se procesa
    status = st.status(label="Buscando información relevante...", expanded=True)

    # --- Preparar Prompts ---
    # Nombres de archivos para el prompt
    manual_filename = os.path.basename(pdf_file_path) if pdf_file_path else "Manual del Empleado"
    codigo_filename = os.path.basename(codigo_trabajo_path) if codigo_trabajo_path else "Código del Trabajo"

    system_prompt = f"""Eres un asistente experto de RRHH para la 'Asociación Pro Desarrollo Comunal del Patio, Inc.'. Tu cultura es FORMAL pero INNOVADORA y cercana.
    Tu base de conocimiento principal son dos documentos:
    1. El Manual del Empleado ({manual_filename})
    2. El Código del Trabajo de Chile ({codigo_filename})

    Instrucciones Generales:
    - Responde de manera ejecutiva, precisa y asertiva.
    - Basa tus respuestas ESTRICTAMENTE en el contexto proporcionado de los documentos. NO inventes información ni uses conocimiento externo no validado por el contexto.
    - Si el contexto contiene la respuesta, explícala claramente, citando la fuente (Manual o Código Trabajo) y aproximadamente la página si es posible (indicado en el contexto como 'Pág. ~X'). Resume y adapta la información para que sea fácil de entender por un empleado. No te limites a copiar el texto del contexto.
    - Si la pregunta NO puede responderse con el contexto proporcionado o si no se encontró contexto relevante, indícalo CLARAMENTE diciendo algo como "La información específica para responder a tu pregunta no se encontró en el contexto de los documentos proporcionados (Manual y Código del Trabajo)." o "No encontré detalles sobre eso en los documentos disponibles."
    - Evita frases como "Basado en el contexto proporcionado..." al inicio de cada respuesta. Ve directo al grano.
    - Mantén un tono profesional pero amable.
    - No respondas a preguntas fuera del ámbito de RRHH o de los documentos proporcionados. Si te preguntan algo no relacionado, indica que solo puedes ayudar con temas del Manual del Empleado o el Código del Trabajo.
    """

    prompt_con_contexto_template = """{contexto}

---
Pregunta del Usuario: {pregunta}
---

Instrucción Específica: Responde a la pregunta basándote ESTRICTAMENTE en el contexto anterior. Cita la fuente (Manual/Código Trabajo) y página aproximada si la información proviene del contexto. Si la respuesta no está en el contexto, indícalo claramente."""

    prompt_sin_contexto_template = """Pregunta del Usuario: {pregunta}
---
Instrucción Específica: No se encontró contexto relevante en los documentos para esta pregunta. Responde indicando claramente que la información no está disponible en el Manual del Empleado ni en el Código del Trabajo que tienes cargados. No intentes adivinar ni usar conocimiento externo."""

    final_user_prompt = ""
    ref_info_for_state = None # Información de referencia para mostrar en la UI

    try:
        # 1. Buscar Contexto
        if not DOC_LOAD_ERROR:
            status.update(label="Buscando contexto en los documentos...", state="running", expanded=True)
            contexto, paginas_fuente_data = find_relevant_context(user_question_clean, top_n=3) # Aumentado a 3 chunks
        else:
            contexto, paginas_fuente_data = None, []
            st.info("Búsqueda de contexto omitida debido a errores previos en la carga de documentos.")

        # 2. Preparar Prompt Final y Referencias
        if contexto and paginas_fuente_data:
            # Limitar tamaño del contexto para no exceder límites del LLM
            max_context_len = 3500 # Ajustable
            if len(contexto) > max_context_len:
                contexto = contexto[:max_context_len] + "\n\n[Contexto truncado por longitud]"
                st.info("El contexto encontrado era muy largo y fue recortado. La respuesta podría ser parcial. Intenta una pregunta más específica si es necesario.")

            final_user_prompt = prompt_con_contexto_template.format(contexto=contexto, pregunta=user_question_clean)
            ref_info_for_state = paginas_fuente_data # Guardamos la lista de dicts {'fuente': '...', 'pagina': ...}
            status.update(label=f"Contexto encontrado. Consultando a {DISPLAY_MODEL_NAME}...", state="running")
        else:
            # Si no hay contexto o hubo error previo
            print("Advertencia: No se encontró contexto relevante o hubo error previo.")
            final_user_prompt = prompt_sin_contexto_template.format(pregunta=user_question_clean)
            ref_info_for_state = "Contexto no encontrado" # Marcador simple
            status.update(label=f"No se encontró contexto. Consultando a {DISPLAY_MODEL_NAME}...", state="running")

        # 3. Llamar al LLM
        status.update(label=f"Generando respuesta con {DISPLAY_MODEL_NAME}...", state="running")
        with st.spinner(f"Consultando al modelo IA ({DISPLAY_MODEL_NAME})..."):
             llm_response_raw = get_llm_response(system_prompt, final_user_prompt, LLM_MODEL_NAME, LLM_API_URL)

        # 4. Procesar Respuesta del LLM (ya se hace limpieza básica dentro de get_llm_response)
        respuesta_final = llm_response_raw # Usar la respuesta ya (algo) limpia

        # 5. Añadir respuesta del bot a la historia
        # Asegurarse de que ref_info_for_state sea serializable si se guarda directamente
        st.session_state.chat.append({'role': 'bot', 'msg': respuesta_final, 'ref': ref_info_for_state})
        status.update(label="Respuesta recibida.", state="complete", expanded=False)

    except Exception as e:
        print(f"Error grave en process_user_input: {e}")
        print(traceback.format_exc())
        st.error(f"Ocurrió un error inesperado al procesar tu pregunta: {e}")
        # Añadir mensaje de error al chat
        error_message = f"Lo siento, ocurrió un error interno ({type(e).__name__}) al intentar procesar tu solicitud. Por favor, inténtalo de nuevo más tarde o contacta al administrador."
        st.session_state.chat.append({'role': 'bot', 'msg': error_message, 'ref': 'Error Proceso'})
        status.update(label="Error", state="error", expanded=True)

# --- === App Execution Starts Here === --

st.markdown(f"<div style='text-align:center; color: #6b3e26; font-size:1.25rem; margin-bottom:1.5rem;'>Asistente Virtual de Recursos Humanos</div>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color: #546e7a; font-size:1.05rem;'>Consulta información del Manual del Empleado ({os.path.basename(pdf_file_path)}) y el Código del Trabajo-Chile ({os.path.basename(codigo_trabajo_path)}) usando IA local ({DISPLAY_MODEL_NAME}).</p>", unsafe_allow_html=True)

# --- Initialize Chat State ---
if "chat" not in st.session_state:
    st.session_state["chat"] = []
    if DOC_LOAD_ERROR:
         # Mensaje inicial si hubo error cargando docs
         st.session_state["chat"].append({
            'role': 'bot',
            'msg': f'⚠️ **Error Crítico:** No se pudieron cargar o procesar correctamente uno o ambos documentos fuente ({os.path.basename(pdf_file_path)}, {os.path.basename(codigo_trabajo_path)}). Mis capacidades estarán limitadas o seré incapaz de responder preguntas basadas en ellos.',
            'ref': 'Error Carga Docs'
        })
    elif CHUNKS_GLOBAL is None or VECTORIZER_GLOBAL is None:
         # Mensaje si algo falló después de la carga inicial pero antes del TF-IDF
         st.session_state["chat"].append({
            'role': 'bot',
            'msg': f'⚠️ **Advertencia:** Hubo un problema al preparar los datos de los documentos. Es posible que no pueda encontrar información relevante.',
            'ref': 'Error Preparación Datos'
        })
    else:
        # Mensaje inicial normal
        st.session_state["chat"].append({
            'role': 'bot',
            'msg': '¡Hola! Soy tu asistente virtual de RRHH. ¿En qué puedo ayudarte respecto al Manual del Empleado o el Código del Trabajo?',
            'ref': 'Inicial'
        })


# --- Display Existing Chat Messages ---
st.divider()
chat_placeholder = st.container() # Contenedor para mensajes

with chat_placeholder:
    if "chat" in st.session_state:
        for i, turno in enumerate(st.session_state["chat"]):
            try:
                is_user = turno['role'] == 'user'
                css_class = "chat-msg-user" if is_user else "chat-msg-bot"
                avatar = "🧑‍💻" if is_user else "💼"
                speaker_name = "Tú" if is_user else "Asistente RRHH"

                cols = st.columns([0.18, 0.64, 0.18])
                if is_user:
                    with cols[1]:
                        st.markdown(f'''
                        <div style="background:rgba(247,178,103,0.75); color:#6b3e26; border-radius:12px 12px 4px 12px; font-size:1.15rem; font-weight:500; box-shadow:0 6px 22px 0 rgba(184,92,56,0.08); border:1.5px solid rgba(249,219,189,0.45); word-break:break-word; backdrop-filter:blur(2px); text-align:left; padding:18px 28px; margin:8px 0 8px auto; white-space:pre-line; display:inline-block; max-width:80%; float:right;">
                        <b>{avatar} {speaker_name}</b>:<br>{turno.get("msg", "*Mensaje no disponible*")}
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    with cols[1]:
                        st.markdown(f'''
                        <div style="background:rgba(255,229,217,0.75); color:#7d4f2a; border-radius:12px 12px 12px 4px; font-size:1.15rem; font-weight:500; box-shadow:0 6px 22px 0 rgba(231,198,169,0.10); border:1.5px solid rgba(249,219,189,0.35); word-break:break-word; backdrop-filter:blur(2px); text-align:left; padding:18px 28px; margin:8px auto 8px 0; white-space:pre-line; display:inline-block; max-width:80%; float:left;">
                        <b>{avatar} {speaker_name}</b>:<br>{turno.get("msg", "*Mensaje no disponible*")}
                        </div>
                        ''', unsafe_allow_html=True)

                # Mostrar Referencias/contexto para respuestas del bot
                if not is_user:
                    ref_info = turno.get('ref')
                    ref_html = ""
                    if isinstance(ref_info, list) and ref_info: # Lista de dicts {'fuente': '...', 'pagina': ...}
                         # Agrupar por fuente
                         refs_by_source = {}
                         for ref_item in ref_info:
                             source = ref_item.get('fuente', 'Desconocida')
                             page = ref_item.get('pagina', '?')
                             if source not in refs_by_source:
                                 refs_by_source[source] = []
                             if page not in refs_by_source[source]: # Evitar páginas duplicadas por fuente
                                refs_by_source[source].append(str(page))

                         ref_parts = []
                         for source, pages in refs_by_source.items():
                             pages_str = ", ".join(sorted(pages, key=int))
                             ref_parts.append(f"{source} (pág. ~{pages_str})")
                         if ref_parts:
                             ref_html = f'<div class="chat-ref">Fuentes: {"; ".join(ref_parts)}</div>'

                    elif isinstance(ref_info, str) and ref_info not in ['Inicial', 'Error Carga Docs', 'Error Preparación Datos', 'Error Proceso', 'Contexto no encontrado']:
                        # Caso de string simple (podría ser un remanente o un caso especial)
                        ref_html = f'<div class="chat-ref">Referencia: {ref_info}</div>'
                    elif ref_info == "Contexto no encontrado":
                        ref_html = f'<div class="chat-ref">Nota: No se encontró información específica en los documentos.</div>'
                    elif ref_info == "Error Carga Docs" or ref_info == "Error Preparación Datos":
                         ref_html = f'<div class="chat-ref" style="color: red;">{ref_info}</div>' # Error destacado
                    elif ref_info == "Error Proceso":
                        ref_html = f'<div class="chat-ref" style="color: orange;">Error durante el procesamiento.</div>'

                    if ref_html:
                        # Mostrar la referencia alineada con la burbuja del bot
                        if is_user:
                            with cols[1]:
                                st.markdown(ref_html, unsafe_allow_html=True)
                        else:
                            with cols[1]:
                                st.markdown(ref_html, unsafe_allow_html=True)

            except Exception as display_e:
                print(f"Error al mostrar mensaje {i}: {display_e}")
                print(traceback.format_exc())
                st.error(f"Error al mostrar un mensaje del chat (índice {i}).")

# --- Chat Input Widget ---
# El input se queda al final por defecto
st.chat_input(
    f"Escribe tu pregunta sobre el Manual o Código aquí...",
    key="chat_input_key", # Clave para acceder al valor en session_state
    on_submit=process_user_input,
    disabled=DOC_LOAD_ERROR # Deshabilitar input si hubo error crítico al cargar docs
)

if DOC_LOAD_ERROR:
     st.warning("El chat está deshabilitado porque no se pudieron cargar los documentos necesarios.", icon="⚠️")

# --- Footer ---
st.divider()
st.caption(f"Asistente RRHH v2.0 | Potenciado por: {DISPLAY_MODEL_NAME} local | Documentos: {os.path.basename(pdf_file_path)}, {os.path.basename(codigo_trabajo_path)}")