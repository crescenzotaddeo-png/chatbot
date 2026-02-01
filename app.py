import re, requests, tempfile
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import streamlit as st
from docx import Document
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#DOCX -> sections -> chunks
@dataclass
class Chunk:
    id: int
    section: str
    text: str

def extract_sections(docx_path: str) -> List[Tuple[str, str]]:
    doc = Document(docx_path)
    out: List[Tuple[str, str]] = []
    current = "Documento"
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if not t:
            continue
        #Righe in MAIUSCOLO come intestazioni
        if len(t) < 90 and t.isupper():
            current = t
            continue
        out.append((current, t))
    return out

def chunk_text(sections: List[Tuple[str, str]], max_chars: int = 2200, overlap: int = 300) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not sections:
        return chunks

    sec = sections[0][0]
    buf = ""
    cid = 0

    def flush():
        nonlocal buf, cid
        txt = buf.strip()
        if txt:
            chunks.append(Chunk(cid, sec, txt))
            cid += 1
        buf = ""

    for s, p in sections:
        if s != sec and buf:
            flush()
            sec = s
        if len(buf) + len(p) + 2 <= max_chars:
            buf += ("\n" + p)
        else:
            flush()
            #Riusa coda del chunk precedente
            if overlap > 0 and chunks:
                tail = chunks[-1].text[-overlap:]
                buf = tail + "\n" + p
            else:
                buf = p

    flush()
    return chunks



#Retrieval (TF-IDF) + soglia
def build_index(chunks: List[Chunk]):
    vec = TfidfVectorizer()
    X = vec.fit_transform([c.text for c in chunks])
    return vec, X

def retrieve(vec, X, chunks: List[Chunk], query: str, k: int = 5, min_sim: float = 0.10) -> List[Chunk]:
   
#Restituisce i top-k chunk solo se la similarità massima supera una soglia.
#Se è sotto soglia -> query fuori contesto -> ritorna [].

    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]

    best = float(np.max(sims)) if sims.size else 0.0
    if best < min_sim:
        return []

    top = np.argsort(-sims)[:k]
    return [chunks[i] for i in top]



#Ollama API
def ask_ollama(model: str, prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 2048,
            "num_predict": 600,
            "temperature": 0.2
        }
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def format_history(messages: List[Dict[str, Any]], max_turns: int = 6, max_chars: int = 1200) -> str:
    tail = messages[-(max_turns * 2):]
    lines = []
    for m in tail:
        role = "Utente" if m["role"] == "user" else "Assistente"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    text = "\n".join(lines).strip()
    return text[-max_chars:]


def build_prompt(question: str, retrieved: List[Chunk], history_text: str) -> str:
    context = "\n\n---\n\n".join([f"[Sezione: {c.section} | Chunk: {c.id}]\n{c.text}" for c in retrieved])

    prompt = f"""
Sei "Paper Tutor": un assistente che risponde SOLO usando il CONTENUTO del documento fornito nel contesto.
Regole (obbligatorie):
1) Usa la CRONOLOGIA solo per capire riferimenti (es. "questo", "quello di prima"), ma NON come fonte.
2) Rispondi esclusivamente con informazioni presenti nel CONTESTO del documento.
3) Se nel contesto NON c'è la risposta, scrivi: "Nel documento non trovo questa informazione."
4) Alla fine aggiungi una riga: Fonti: ... indicando Sezione e Chunk usati.
5) Non inventare definizioni, nomi, date o bibliografia non presenti nel contesto.

CRONOLOGIA (per continuità, NON è fonte):
{history_text if history_text else "(vuota)"}

CONTESTO (estratti dal documento):
{context}

DOMANDA:
{question}

RISPOSTA:
""".strip()
    return prompt


#Streamlit UI (CHAT)

st.set_page_config(page_title="Paper Tutor — Chat AI Locale", layout="wide")
st.title("🤖 - ChatBot")

with st.sidebar:
    model = st.selectbox("Modello Ollama", ["qwen2.5:3b", "phi3:mini"])
    top_k = st.slider("Numero estratti usati (top-k)", 2, 10, 5)
    max_turns = st.slider("Memoria conversazione (turni)", 2, 12, 6)
    min_sim = st.slider("Soglia pertinenza (min_sim)", 0.00, 0.40, 0.11, 0.01)
    st.caption("Se la similarità massima è sotto soglia, la domanda è considerata fuori contesto.")
    st.divider()
    if st.button("🧹 Reset chat"):
        st.session_state.pop("messages", None)
        st.rerun()

uploaded = st.file_uploader("Carica il tuo paper (DOCX)", type=["docx"])

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_data(show_spinner=False)
def build_doc_index(doc_bytes: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as f:
        f.write(doc_bytes)
        path = f.name
    sections = extract_sections(path)
    chunks = chunk_text(sections)
    vec, X = build_index(chunks)
    return chunks, vec, X

if not uploaded:
    st.info("Carica un file DOCX per iniziare.")
    st.stop()

chunks, vec, X = build_doc_index(uploaded.getvalue())

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_q = st.chat_input("Scrivi una domanda sul documento…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.write(user_q)

    retrieved = retrieve(vec, X, chunks, user_q, k=top_k, min_sim=min_sim)

    #BLOCCO FUORI CONTESTO: non chiamare Ollama
    if not retrieved:
        ans = "La domanda non è pertinente al contenuto del documento."
        with st.chat_message("assistant"):
            st.write(ans)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.stop()

    history_text = format_history(st.session_state.messages[:-1], max_turns=max_turns)
    prompt = build_prompt(user_q, retrieved, history_text)

    with st.chat_message("assistant"):
        with st.spinner("Sto ragionando sul documento…"):
            try:
                ans = ask_ollama(model, prompt)
            except Exception as e:
                ans = f"Errore chiamando Ollama: {e}"

        st.write(ans)

        with st.expander("📌 Fonti (estratti usati dal documento)", expanded=False):
            for c in retrieved:
                st.markdown(f"**{c.section} — Chunk {c.id}**")
                st.write(c.text)
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": ans})
