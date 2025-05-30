# app.py
import os
import re
import json
import traceback
import requests
import PyPDF2
import pickle
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from flask import Flask, render_template, request, jsonify, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# --- Variables de backend LLM ---
LLM_API_URL = os.environ.get("LLM_API_URL", "http://127.0.0.1:1234/v1/chat/completions")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "llama4-dolphin-8b")
try:
    name_parts = re.split(r'[:/-]', LLM_MODEL_NAME)
    DISPLAY_MODEL_NAME = name_parts[-1].replace('-', ' ').title()
except Exception:
    DISPLAY_MODEL_NAME = LLM_MODEL_NAME

# --- Rutas de los PDFs y caché ---
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_file_path = os.path.join(script_dir, "Prueba4.pdf")
codigo_trabajo_path = os.path.join(script_dir, "Código del Trabajo-Chile.pdf")

# --- Rutas de caché ---
CACHE_DIR = os.path.join(script_dir, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CHUNKS_CACHE = os.path.join(CACHE_DIR, "chunks.pkl")
TFIDF_CACHE = os.path.join(CACHE_DIR, "tfidf.pkl")
CTX_CACHE = os.path.join(CACHE_DIR, "ctx_cache.pkl")

# --- Carpeta para guardar conversaciones históricas ---
CONV_DIR = os.path.join(script_dir, "conversations")
os.makedirs(CONV_DIR, exist_ok=True)

# Verificación clara de existencia de los PDFs
DOC_LOAD_ERROR = False
missing_files = []
if not os.path.exists(pdf_file_path):
    missing_files.append("Prueba4.pdf")
    DOC_LOAD_ERROR = True
if not os.path.exists(codigo_trabajo_path):
    missing_files.append("Código del Trabajo-Chile.pdf")
    DOC_LOAD_ERROR = True
if DOC_LOAD_ERROR:
    print("No se encontraron los siguientes archivos PDF requeridos:")
    for f in missing_files:
        print(" -", f)
else:
    print("Todos los archivos PDF requeridos están presentes.")

# --- Flags y caches ---
CHUNKS_GLOBAL = None
CORPUS_GLOBAL = None
VECTORIZER_GLOBAL = None
CORPUS_VECS_GLOBAL = None

CHUNK_SIZE = 600
CHUNK_OVERLAP = 75

def save_cache(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)

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
            print(traceback.format_exc())
    if not all_chunks and not DOC_LOAD_ERROR:
        print("No se extrajeron chunks de texto.")
    return all_chunks

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
        print(traceback.format_exc())
        return None, None, None

# --- Cache de contexto relevante para preguntas repetidas ---
def get_ctx_cache():
    try:
        if os.path.exists(CTX_CACHE):
            return load_cache(CTX_CACHE)
        else:
            return {}
    except Exception:
        return {}

def save_ctx_cache(cache):
    try:
        save_cache(cache, CTX_CACHE)
    except Exception:
        pass

def find_relevant_context(pregunta: str, top_n: int = 1) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if DOC_LOAD_ERROR or not CHUNKS_GLOBAL or not pregunta or not pregunta.strip() or VECTORIZER_GLOBAL is None or CORPUS_VECS_GLOBAL is None:
        print("Contexto no buscado: Faltan datos o hubo error previo.")
        return None, []
    pregunta_key = pregunta.strip().lower()
    ctx_cache = get_ctx_cache()
    if pregunta_key in ctx_cache:
        return ctx_cache[pregunta_key], []
    try:
        q_vec = VECTORIZER_GLOBAL.transform([pregunta])
        sims = cosine_similarity(q_vec, CORPUS_VECS_GLOBAL)[0]
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        threshold = 0.05
        top_idxs = [i for i, s in ranked if s > threshold][:top_n]
        if not top_idxs:
            ctx_cache[pregunta_key] = (None, [])
            save_ctx_cache(ctx_cache)
            return None, []
        ctx = "Contexto relevante:\n\n"
        seen = set()
        for idx in top_idxs:
            chunk = CHUNKS_GLOBAL[idx]
            if chunk["texto"] not in seen:
                ctx += f"Fuente: {chunk['fuente']}, Pág. ~{chunk['pagina']}\n"
                ctx += f'"{chunk["texto"]}"\n\n'
                seen.add(chunk["texto"])
        ctx_cache[pregunta_key] = (ctx.strip(), [])
        save_ctx_cache(ctx_cache)
        print(f"Contexto encontrado ({len(top_idxs)} chunks) para: '{pregunta[:50]}...'")
        return ctx.strip(), []
    except Exception as e:
        print(f"Error durante la búsqueda de contexto TF-IDF: {e}")
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

# --- Guardar histórico de conversaciones ---
def save_current_conversation():
    if "chat" in session and session["chat"]:
        conv_id = session.get("conv_id")
        if not conv_id:
            # Busca la primera pregunta del usuario
            first_user_msg = next((m for m in session["chat"] if m["role"] == "user"), None)
            if first_user_msg:
                # Toma las primeras 6 palabras de la pregunta
                pregunta = first_user_msg["content"].strip().replace('\n', ' ')
                pregunta = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚüÜñÑ ]', '', pregunta)
                pregunta = '_'.join(pregunta.split()[:6])
            else:
                pregunta = "sin_pregunta"
            fecha = datetime.now().strftime("%Y%m%d_%H%M")
            conv_id = f"{fecha}_{pregunta}_{str(uuid.uuid4())[:6]}"
            session["conv_id"] = conv_id
        path = os.path.join(CONV_DIR, f"{conv_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(session["chat"], f, ensure_ascii=False, indent=2)

def list_conversations():
    files = [f for f in os.listdir(CONV_DIR) if f.endswith(".json")]
    files.sort(reverse=True)
    return files

def load_conversation(conv_id):
    path = os.path.join(CONV_DIR, f"{conv_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            session["chat"] = json.load(f)
            session["conv_id"] = conv_id

# --- Manejo de mensajes y guardado de histórico ---
def handle_new_message(user_text: str):
    ts = datetime.now().strftime("%H:%M")
    if "chat" not in session:
        session["chat"] = []
    session["chat"].append({
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
    session["chat"].append({
        "role": "assistant",
        "content": content,
        "timestamp": bot_ts
    })
    save_current_conversation()  # Guarda cada vez

# --- Carga de PDFs e índices optimizada ---
def load_all_with_cache():
    global CHUNKS_GLOBAL, CORPUS_GLOBAL, VECTORIZER_GLOBAL, CORPUS_VECS_GLOBAL, DOC_LOAD_ERROR
    try:
        if os.path.exists(CHUNKS_CACHE) and os.path.exists(TFIDF_CACHE):
            CHUNKS_GLOBAL = load_cache(CHUNKS_CACHE)
            CORPUS_GLOBAL, VECTORIZER_GLOBAL, CORPUS_VECS_GLOBAL = load_cache(TFIDF_CACHE)
            print("Cache de PDFs y TF-IDF cargado.")
        else:
            CHUNKS_GLOBAL = load_and_chunk_pdfs(pdf_file_path, codigo_trabajo_path)
            if CHUNKS_GLOBAL:
                save_cache(CHUNKS_GLOBAL, CHUNKS_CACHE)
                CORPUS_GLOBAL, VECTORIZER_GLOBAL, CORPUS_VECS_GLOBAL = precalculate_tfidf_vectors(CHUNKS_GLOBAL)
                save_cache((CORPUS_GLOBAL, VECTORIZER_GLOBAL, CORPUS_VECS_GLOBAL), TFIDF_CACHE)
                print("PDFs procesados y cacheados.")
            else:
                DOC_LOAD_ERROR = True
    except Exception as e:
        print("Error cargando o guardando el cache:", e)
        DOC_LOAD_ERROR = True

if not DOC_LOAD_ERROR:
    load_all_with_cache()

@app.route('/')
def index():
    if DOC_LOAD_ERROR:
        missing = '<br>'.join(missing_files) if missing_files else 'Desconocido'
        return render_template('index.html', chat=[], error=f"No se encontraron los siguientes archivos PDF requeridos:<br>{missing}")
    if "chat" not in session:
        session["chat"] = [
            {
                "role": "assistant",
                "content": "¡Hola! Soy Aura ✨. ¿Consultas sobre el Manual o Código del Trabajo?",
                "timestamp": datetime.now().strftime("%H:%M")
            }
        ]
    return render_template('index.html', chat=session["chat"], error=None)

@app.route('/send_message', methods=['POST'])
def send_message():
    user_text = request.form['message']
    handle_new_message(user_text)
    session.modified = True  # Fuerza a Flask a guardar la sesión
    return jsonify(session["chat"])

# --- Nuevas rutas para el histórico ---
@app.route('/conversations')
def conversations():
    files = list_conversations()
    # Devuelve lista de IDs y fechas legibles
    return jsonify([
        {"id": f[:-5], "name": f.replace(".json", "").replace("_", " ")}
        for f in files
    ])

@app.route('/load_conversation/<conv_id>')
def load_conv(conv_id):
    load_conversation(conv_id)
    return jsonify(session["chat"])

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    session.pop("chat", None)
    session.pop("conv_id", None)
    return jsonify({"ok": True})

if __name__ == '__main__':
    app.run(debug=False)