# Paper Tutor — Chatbot AI locale (Ollama) collegato a un DOCX

Questo progetto è una web app in Python (Streamlit) che permette di **conversare** con un documento DOCX.  
Il sistema usa un approccio **RAG (Retrieval-Augmented Generation)**: recupera gli elementi più pertinenti alla richiesta dell'utente dal documento
e chiede ad un modello LLM **locale** (via Ollama) di rispondere **solo** utilizzando i contenuti del documento.

## Funzionalità principali
- Caricamento di un documento **.docx**
- Segmentazione del testo in *chunk* (porzioni) con metadati di sezione
- Indicizzazione e retrieval tramite **TF-IDF + similarità coseno**
- Chat conversazionale con memoria (Streamlit `session_state`)
- Modello LLM locale tramite **Ollama** (nessuna API a pagamento)
- Protezione “fuori contesto”: se la query non è pertinente al documento, il bot risponde che non è pertinente
- Visualizzazione delle **fonti** (estratti usati)

## Requisiti
- Python 3.10+ (consigliato)
- Ollama installato: https://ollama.com
- Un modello scaricato (es. `qwen2.5:3b` oppure `phi3:mini`)

## Installazione (Windows / macOS / Linux)

### 1) Installa le dipendenze Python
```bash
pip install -r requirements.txt
```
### 2) Scarica un modello Ollama (scegline uno)
Consigliato:
```bash
ollama pull qwen2.5:3b
```
### 3) Avvia l'app
```bash
streamlit run app.py
```

