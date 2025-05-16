# app.py
import os
import re
import json
import traceback
import requests
import PyPDF2
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# --- Variables de backend LLM ---
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemma-3-1b-it-qat")
try:
    name_parts = re.split(r'[:/-]', LLM_MODEL_NAME)
    DISPLAY_MODEL_NAME = name_parts[-1].replace('-', ' ').title()
except Exception:
    DISPLAY_MODEL_NAME = LLM_MODEL_NAME

# --- Rutas de los PDFs ---
script_dir = os.path.dirname(os.path.abspath(__file__))
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
    DOC_LOAD_ERROR = True

if not os.path.exists(codigo_trabajo_path):
    DOC_LOAD_ERROR = True

CHUNK_SIZE = 600
CHUNK_OVERLAP = 75

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
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
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

# --- Carga de PDFs e índices ---
if not DOC_LOAD_ERROR:
    CHUNKS_GLOBAL = load_and_chunk_pdfs(pdf_file_path, codigo_trabajo_path)
    if CHUNKS_GLOBAL:
        CORPUS_GLOBAL, VECTORIZER_GLOBAL, CORPUS_VECS_GLOBAL = precalculate_tfidf_vectors(CHUNKS_GLOBAL)
        if CORPUS_VECS_GLOBAL is None:
            DOC_LOAD_ERROR = True
    else:
        DOC_LOAD_ERROR = True

@app.route('/')
def index():
    if DOC_LOAD_ERROR:
        return render_template('index.html', chat=[], error="No se pudieron cargar los documentos PDF requeridos. Por favor verifica los archivos.")
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
    return jsonify(session["chat"])

if __name__ == '__main__':
    app.run(debug=True)

#Código de `templates