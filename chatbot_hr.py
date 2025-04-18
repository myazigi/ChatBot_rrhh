import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import os
import re # Para ayudar en la limpieza de texto si es necesario

# --- Configuración Mejorada ---
PDF_PATH = "Prueba4.pdf"
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

# --- Parámetros de Chunking ---
# Tamaño objetivo de cada fragmento de texto (en caracteres)
CHUNK_SIZE = 700
# Solapamiento entre fragmentos consecutivos (en caracteres)
CHUNK_OVERLAP = 100

# --- Parámetros de Búsqueda ---
# Cuántos chunks relevantes devolver como máximo
TOP_K = 3
# Umbral de similitud MÍNIMO para considerar un chunk relevante
SIMILARITY_THRESHOLD = 0.5 # Puedes ajustar esto. Quizás bajarlo un poco al usar chunks más pequeños.

# --- Funciones Auxiliares ---

@st.cache_resource
def load_model(model_name):
    """Carga el modelo de Sentence Transformer."""
    print(f"Intentando cargar el modelo: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("Modelo cargado exitosamente.")
        return model
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo '{model_name}'. Verifica la instalación y el nombre del modelo.")
        st.exception(e)
        st.stop()

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Divide un texto largo en fragmentos más pequeños con solapamiento."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # Avanza para el siguiente chunk, retrocediendo por el overlap
        # Asegura que no haya un bucle infinito si overlap >= size
        next_start = start + chunk_size - chunk_overlap
        if next_start <= start: # Previene bucle si overlap es muy grande o size pequeño
             next_start = start + 1 # Avanza al menos 1 caracter
        start = next_start

    # Eliminar chunks muy pequeños que puedan quedar al final si son solo overlap
    # return [chk for chk in chunks if len(chk.strip()) > chunk_overlap / 2] # Opcional
    return chunks


@st.cache_data # Cache basado en el contenido del PDF y parámetros de chunking
def load_and_process_pdf_chunked(pdf_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Lee el PDF, extrae texto, lo limpia y lo divide en chunks."""
    print(f"Procesando PDF y dividiendo en chunks: {pdf_path} (size={chunk_size}, overlap={chunk_overlap})")
    if not os.path.exists(pdf_path):
        st.error(f"Error: El archivo '{pdf_path}' no se encuentra.")
        return [], []

    doc_chunks_data = []
    all_texts = []

    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            st.warning(f"El PDF '{pdf_path}' parece estar vacío.")
            doc.close()
            return [], []

        for page_num, page in enumerate(doc):
            page_text = page.get_text("text", sort=True)
            # Limpieza básica (opcional, ajustar según necesidad)
            page_text = re.sub(r'\s+', ' ', page_text).strip() # Reemplaza múltiples espacios/saltos con uno solo

            if page_text:
                page_chunks = chunk_text(page_text, chunk_size, chunk_overlap)
                for i, chunk in enumerate(page_chunks):
                    if chunk.strip(): # Asegurarse que el chunk no esté vacío
                        doc_chunks_data.append({
                            "text": chunk.strip(),
                            "page": page_num + 1,
                            # Podrías añadir inicio/fin del chunk si es útil
                            # "chunk_index_on_page": i
                        })
                        all_texts.append(chunk.strip())
            else:
                 print(f"Página {page_num + 1} sin texto extraíble.")

        doc.close()
        print(f"PDF procesado. {len(doc_chunks_data)} chunks generados.")

        if not doc_chunks_data:
             st.warning(f"No se pudieron generar chunks de texto desde '{pdf_path}'. ¿El PDF tiene texto seleccionable?")
             return [], []

        return doc_chunks_data, all_texts

    except Exception as e:
        st.error(f"Error al procesar/chunkear el PDF '{pdf_path}'.")
        st.exception(e)
        return [], []

# Usamos _model como argumento para que el cache de Streamlit se invalide si el modelo cambia
@st.cache_data
def get_pdf_embeddings(_model, pdf_texts):
    """Genera embeddings para los chunks de texto del PDF."""
    if not pdf_texts:
        st.warning("No hay textos (chunks) para generar embeddings.")
        return None
    if _model is None:
        st.error("Modelo no cargado, no se pueden generar embeddings.")
        return None

    print(f"Generando embeddings para {len(pdf_texts)} chunks...")
    try:
        # Considerar usar show_progress_bar=True para chunks largos
        embeddings = _model.encode(pdf_texts, convert_to_tensor=True, show_progress_bar=True)
        print("Embeddings generados.")
        return embeddings
    except Exception as e:
        st.error("Error al generar embeddings para los chunks.")
        st.exception(e)
        return None

# Cambiado para devolver múltiples chunks relevantes
def find_relevant_chunks(query, _model, pdf_embeddings, pdf_chunks_data, top_k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    """Encuentra los K chunks más relevantes para la consulta que superan el umbral."""
    if _model is None or pdf_embeddings is None or not pdf_chunks_data:
        st.warning("Faltan datos (modelo, embeddings o chunks) para la búsqueda.")
        return [] # Devuelve lista vacía

    print(f"Buscando top-{top_k} chunks para: '{query}' (umbral: {threshold})")
    try:
        query_embedding = _model.encode(query, convert_to_tensor=True)

        # Calcular similitudes
        cosine_scores = util.pytorch_cos_sim(query_embedding, pdf_embeddings)[0]

        # Obtener los top_k resultados (índices y scores)
        # Asegurarse de que top_k no sea mayor que el número de chunks
        actual_k = min(top_k, len(pdf_chunks_data))
        top_results = torch.topk(cosine_scores, k=actual_k)

        relevant_chunks = []
        scores_found = [] # Para depuración o mensajes
        for score, idx in zip(top_results[0], top_results[1]):
            score_item = score.item()
            idx_item = idx.item()
            scores_found.append(score_item) # Guardar todos los K scores

            if score_item >= threshold:
                chunk_data = pdf_chunks_data[idx_item]
                relevant_chunks.append({
                    "text": chunk_data["text"],
                    "page": chunk_data["page"],
                    "score": score_item
                })
                print(f"  - Chunk relevante encontrado (Página {chunk_data['page']}, Score: {score_item:.4f})")
            else:
                # Si el score más alto ya está por debajo del umbral, los demás también lo estarán (ya que topk ordena)
                print(f"  - Score {score_item:.4f} por debajo del umbral {threshold}. Deteniendo búsqueda temprana.")
                break # Optimización: no seguir si ya bajamos del umbral

        if not relevant_chunks:
            print(f"No se encontraron chunks por encima del umbral. Scores más altos: {scores_found}")

        # Devolver la lista de chunks relevantes (puede estar vacía)
        return relevant_chunks, scores_found # Devolvemos scores para mensaje de "no encontrado"

    except Exception as e:
        st.error("Error durante la búsqueda de chunks relevantes.")
        st.exception(e)
        return [], []


# --- Interfaz de Streamlit ---
st.set_page_config(page_title="Chatbot RRHH Avanzado", layout="wide")

st.title("🤖 Asistente Virtual de Recursos Humanos (v2 - Precisión Mejorada)")
st.caption(f"Consultando información en: {os.path.basename(PDF_PATH)} (Fragmentado para mayor precisión)")

# --- Carga y Procesamiento ---
try:
    model = load_model(MODEL_NAME)
    if model:
        pdf_chunks_data, pdf_texts = load_and_process_pdf_chunked(PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
        pdf_embeddings = None
        if pdf_texts:
            pdf_embeddings = get_pdf_embeddings(model, pdf_texts)
        else:
            st.warning("No se generaron embeddings porque no se extrajo texto útil del PDF.")
    else:
        st.error("La carga del modelo falló. El chatbot no puede operar.")
        pdf_chunks_data, pdf_texts, pdf_embeddings = [], [], None

except Exception as main_load_error:
    st.error("Ocurrió un error crítico durante la inicialización.")
    st.exception(main_load_error)
    model, pdf_chunks_data, pdf_texts, pdf_embeddings = None, [], [], None


# --- Área de Chat ---
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "¿Cómo puedo ayudarte con nuestros procedimientos de RRHH hoy?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Usar st.write o st.markdown para permitir formato más rico si es necesario
        st.write(message["content"])

user_query = st.chat_input("Escribe tu consulta detallada aquí...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Generar y mostrar respuesta del asistente
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_container.write("Analizando tu consulta y buscando en el documento...")

        if model is None or pdf_embeddings is None or not pdf_chunks_data:
             final_response = f"Lo siento, no puedo procesar tu consulta en este momento. Hay un problema con la carga del modelo o el procesamiento del documento '{os.path.basename(PDF_PATH)}'."
             response_container.error(final_response)
        else:
            # Buscar los chunks relevantes
            relevant_chunks, all_top_scores = find_relevant_chunks(
                user_query, model, pdf_embeddings, pdf_chunks_data, TOP_K, SIMILARITY_THRESHOLD
            )

            if relevant_chunks:
                # Construir una respuesta unificada
                response_parts = ["Encontré las siguientes secciones que parecen más relevantes para tu consulta:\n"]
                for i, chunk in enumerate(relevant_chunks):
                    response_parts.append(f"--- **Fragmento {i+1} (Página {chunk['page']}, Similitud: {chunk['score']:.2f})** ---")
                    response_parts.append(f"> {chunk['text']}") # Usar blockquote para el texto
                    response_parts.append("\n") # Añadir espacio

                response_parts.append(f"*Fuente: {os.path.basename(PDF_PATH)}*")
                final_response = "\n".join(response_parts)
                response_container.info(final_response) # Usar info para destacar

            else:
                # Mensaje más informativo si no se encontró nada útil
                max_score_info = ""
                if all_top_scores:
                    max_score_info = f" La similitud más alta encontrada fue de {max(all_top_scores):.2f}, que está por debajo del umbral requerido ({SIMILARITY_THRESHOLD})."

                final_response = (f"Lo siento, no encontré información suficientemente específica sobre '{user_query}' "
                                  f"en el documento '{os.path.basename(PDF_PATH)}' con el nivel de confianza necesario."
                                  f"{max_score_info} Intenta reformular tu pregunta siendo más específico o usando otras palabras clave.")
                response_container.warning(final_response)

        # Añadir la respuesta final al historial
        st.session_state.messages.append({"role": "assistant", "content": final_response})

# --- Nota Técnica (opcional) ---
# st.sidebar.info(
#     f"**Configuración:** Modelo: {MODEL_NAME}, Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}, Top K: {TOP_K}, Umbral: {SIMILARITY_THRESHOLD}"
# )

# --- Mistral LLM Backend Configuration ---
# Adapt this URL if your Mistral API endpoint is different (e.g., /v1/chat/completions)
MISTRAL_API_URL = "http://localhost:10001/api/generate" # Common Ollama endpoint
# Replace with the actual model name served by your local instance (e.g., "mistral", "mistral:7b")
MISTRAL_MODEL_NAME = "mistral" # <<< CHANGE THIS IF NEEDED

