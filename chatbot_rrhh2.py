import streamlit as st
import os
import requests
import json
import traceback
import tempfile

# --- Streamlit config: ¡Debe ser lo primero! ---
st.set_page_config(page_title="Chatbot RRHH IA", page_icon="💼", layout="wide")

# --- Funciones auxiliares ---
def load_pdf_and_chunks(pdf_file_path):
    # Función para cargar el PDF y dividirlo en fragmentos
    # (Placeholder para tu implementación existente)
    pass

def find_relevant_context(question, chunks, top_n=3):
    # Función para encontrar contexto relevante en el documento
    # (Placeholder para tu implementación existente)
    pass

def get_llm_response(system_prompt, user_prompt, model_name, api_url):
    # Función para obtener respuesta del modelo LLM
    # (Placeholder para tu implementación existente)
    pass

# --- Título y descripción ---
st.markdown(f"<h1 style='text-align:center; color: #263238;'>💼 Chatbot RRHH ({os.getenv('DISPLAY_MODEL_NAME', 'Modelo')})</h1>", unsafe_allow_html=True)
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
for message in st.session_state["chat"]:
    if message["role"] == "bot":
        st.markdown(f"**🤖 Bot:** {message['msg']}")
    else:
        st.markdown(f"**🧑 Usuario:** {message['msg']}")
    if "ref" in message and message["ref"] not in ['Inicial', 'Error Carga', 'Error Proceso', ["Contexto no encontrado"], ["N/A"]]:
        try:
            ref_text = ", ".join(map(str, message["ref"])) if isinstance(message["ref"], list) else str(message["ref"])
            st.markdown(f'<div class="chat-ref">Referencia: {os.path.basename(pdf_file_path) if pdf_file_path else "PDF no cargado"}, pág(s). ~ {ref_text}</div>', unsafe_allow_html=True)
        except Exception as ref_e:
            print(f"Error display ref: {ref_e}")
            st.caption(f"Ref: {message['ref']} (Error)")
    elif message["ref"] == ["Contexto no encontrado"]:
        st.markdown(f'<div class="chat-ref">Nota: No se encontró contexto específico en el documento.</div>', unsafe_allow_html=True)

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

    try:
        with st.spinner("Procesando tu pregunta..."):
            status = st.empty()
            status.update(label="Analizando documento...")
            contexto, paginas_fuente = find_relevant_context(user_question_clean, loaded_chunks, top_n=3)

            prompt_contexto = "Contexto del Manual:\n{contexto}\n\n---\nPregunta del Usuario: {pregunta}\n---\nInstrucciones: Cita páginas si es posible. Si no está en contexto, indícalo. Sé profesional y conciso."
            prompt_sin_contexto = "Pregunta del Usuario: {pregunta}\n---\nInstrucciones: No hay contexto del manual. Si no estás seguro, indícalo. No inventes. Sé profesional y conciso."
            system_prompt = f"Eres un asistente experto de RRHH para la Asociación Pro Desarrollo Comunal del Patio, Inc. Cultura FORMAL E INNOVADORA. Responde basándote *estrictamente* en el contexto del manual ({os.path.basename(pdf_file_path) if pdf_file_path else 'PDF no cargado'}) si se proporciona."

            user_prompt_final = ""
            ref_info_for_state = []
            if contexto:
                user_prompt_final = prompt_contexto.format(contexto=contexto, pregunta=user_question_clean)
                ref_info_for_state = paginas_fuente
                label_text = f"Consultando a {os.getenv('DISPLAY_MODEL_NAME', 'Modelo')}..."
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
                status.update(label=f"Consultando a {os.getenv('DISPLAY_MODEL_NAME', 'Modelo')}...")

            llm_response = get_llm_response(system_prompt, user_prompt_final, os.getenv('LLM_MODEL_NAME', 'default'), os.getenv('LLM_API_URL', 'http://localhost'))
            st.session_state.chat.append({'role': 'bot', 'msg': llm_response, 'ref': ref_info_for_state})
            status.update(label="Respuesta lista.", state="complete", expanded=False)

    except Exception as e:
        print(f"Error en process_user_input: {e}")
        print(traceback.format_exc())
        st.error(f"Ocurrió un error al procesar tu pregunta: {e}")
        error_message = f"Error al procesar: {type(e).__name__}."
        st.session_state.chat.append({'role': 'bot', 'msg': error_message, 'ref': 'Error Proceso'})

# --- Input Box ---
st.text_input("Escribe tu pregunta aquí:", key="chat_input_key", on_change=process_user_input)

# --- Footer ---
st.caption(f"Asistente RRHH v1.9.2 | Potenciado por: {os.getenv('DISPLAY_MODEL_NAME', 'Modelo')} | Documento: {os.path.basename(pdf_file_path) if pdf_file_path else 'PDF no cargado'}")