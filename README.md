# Review Genie: local product assistant with Ollama

`main_agent_file.py` is a small chatbot. It answers questions about Amazon
products using a local FAISS index, and falls back to a web search (Tavily)
when the answer isn't in that index. It runs fully locally using Ollama for
both the chat model and the embeddings, with a Gradio browser interface.

## 1. Install Ollama and pull the models

Install [Ollama](https://ollama.com/download) and make sure it's running
(open the Ollama app, or run `ollama serve` in a terminal). Then pull the
models this app uses:

```sh
ollama pull qwen3.5:2b
ollama pull nomic-embed-text
```

## 2. Set up the Python environment


```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Configure `.env`

Create a `.env` file in this folder:

```dotenv
TAVILY_API_KEY=your_tavily_key
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3.5:2b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
LLM_NUM_CTX=8192
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
```

Keep `.env` out of version control. `TAVILY_API_KEY` is optional — without it,
the app still works, just without web search fallback.

## 4. Build the vector index (only if `Vector store` is missing or the embedding model changed)

The app reads its product data from the `Vector store` folder. Its vectors
must match whatever embedding model `OLLAMA_EMBEDDING_MODEL` is set to. If you
change that model, or `Vector store` is missing, rebuild it:

```sh
python - <<'PY'
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

dir = Path.cwd()
load_dotenv(dir / ".env", override=True)
embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)
source = FAISS.load_local(
    str(dir / "Vector store"),
    embeddings,
    allow_dangerous_deserialization=True,
)
documents = [
    source.docstore.search(document_id)
    for document_id in source.index_to_docstore_id.values()
]
rebuilt = FAISS.from_documents(documents, embeddings)
rebuilt.save_local(str(dir / "Vector store"))
print("Rebuilt Vector store")
PY
```

## 5. Run the app

```sh
python main_agent_file.py
```

Open `http://127.0.0.1:7860` in your browser. Type a question and click
**Submit**. Click **Clear Conversation** to reset the chat history without
restarting the app.

## Troubleshooting

- **"model not found" error**: the model in `.env` (`LLM_MODEL`) hasn't been
  pulled yet. Run `ollama list` to see what's installed, and `ollama pull
  <model>` for what's missing.
- **"Missing Ollama vector index" error**: run the rebuild step above.
- **FAISS dimension error**: `Vector store` was built with a different
  embedding model than `OLLAMA_EMBEDDING_MODEL` is now set to. Rebuild it
  (step 4).
- **"Agent stopped due to iteration limit or time limit"**: the model got
  stuck retrying instead of answering. Try rephrasing the question; if it
  keeps happening on questions that should work, it may need prompt tuning
  in `main_agent.py`.
