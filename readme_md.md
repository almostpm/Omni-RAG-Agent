# Omni-RAG-Agent
A fully autonomous, multimodal RAG (Retrieval-Augmented Generation) agent. 

I read PDFs, OCR images, scrape web links, and execute actions (file expenses, manage a mock shopping cart) via natural language. I am built with LangChain, LangGraph, Gemini 2.5 Flash, FAISS, and Gradio.

Status: 7 build phases (0-6, see Phase Reference below) complete. Offline self-tests passing. Repo: https://github.com/almostpm/Omni-RAG-Agent

## Why Omni-RAG-Agent Exists
Most RAG tutorials hand you a library and hide the mechanics. I exist so you can read every line of a production-grade RAG pipeline and see how chunking, embeddings, FAISS indexing, LangGraph state machines, and SSRF security guards each earn their place. I am fully local (FAISS), highly defensive (Pydantic + strict tool schemas), and complete enough to run.

## What Omni-RAG-Agent Does
| Capability | How I do it |
|---|---|
| **Read documents** | I load PDFs, chunk them (800 chars, 150 overlap), and embed them |
| **Search** | I use FAISS for L2-normalized cosine similarity retrieval |
| **Answer** | I use Gemini 2.5 Flash with a strict score gate (no hallucinations) |
| **Cite** | I enforce Pydantic citation schemas and drop fake page numbers |
| **Multi-PDF** | I support metadata filtering (oversample + Python filter) |
| **Act (Agent)** | I use LangGraph ReAct to decide between tools |
| **OCR** | I use Gemini Vision to read text from uploaded images |
| **Scrape** | I use `trafilatura` to read web links (with SSRF protection) |
| **Mock Actions** | I simulate filing expenses, adding to cart, and placing orders |
| **Web UI** | A Gradio app with PDF/Image/URL tabs |

## Requirements
| Dependency | Version | When needed |
|---|---|---|
| Python | 3.11+ | Always |
| langchain | >= 0.3.0 | Always |
| langgraph | >= 0.2.0 | Agent only |
| faiss-cpu | >= 1.8.0 | Retrieval only |
| gradio | >= 4.0.0 | Web UI only |

## Quick Start
```bash
git clone https://github.com/almostpm/Omni-RAG-Agent.git
cd Omni-RAG-Agent
pip install -r requirements.txt

# Add your Gemini API key to .env
echo "RAG_API_KEY=your_key_here" > .env

# Run the UI
python src/app.py
```
Expected output: `Running on local URL: http://127.0.0.1:7860`

## Install
`requirements.txt` contains the core stack. To run my offline tests (no API key needed), just run the Python files directly:
```bash
python src/tools.py
python src/vision_extract.py --selftest
python src/recipe4_retrieve.py --selftest
```

## Run the Web App
```bash
python src/app.py
```
Open `http://127.0.0.1:7860`. I have three input tabs:

| Tab | What you do |
|---|---|
| PDF | Drag `.pdf` files, click "Index PDF", and wait for the FAISS index to build. |
| Image | Drag a `.png`/`.jpg`, click "Read image", and I will OCR the text. |
| URL | Paste a web link and click "Attach URL" (I will fetch it on demand). |

You can then chat with me. I remember the last 4 turns of our conversation. If you ask me to buy something, I will use mock tools—**I never process real payments or place real orders.**

## Real Output Example
Run this from the project root to test the agent via CLI:
```bash
python src/agent_cli.py --query "Scrape https://example.com and tell me what it says."
```
Output:
```text
[user] Scrape https://example.com and tell me what it says.
[agent] calls tool -> scrape_url({"url": "https://example.com"})
[tool:scrape_url] Page title: Example Domain... Content: This domain is for use in illustrative examples...
[agent] The page at example.com is a standard placeholder domain used for documentation...
```

## Architecture
`INPUT -> INGEST -> EMBED -> INDEX -> RETRIEVE -> GENERATE -> AGENT -> ACT`

This chain shows my pipeline order. It is not a strict dependency graph.

| Layer | File | Depends On |
|---|---|---|
| **CONFIG** | `config.py` | Nothing (loads `.env`) |
| **INGEST** | `recipe1_ingest.py` | `config` |
| **EMBED** | `recipe2_embed.py` | `config`, Gemini API |
| **INDEX** | `recipe3_build_index.py` | `config`, FAISS |
| **RETRIEVE** | `recipe4_retrieve.py` | `config`, FAISS |
| **QA** | `qa_chain.py` | `recipe4`, Gemini API |
| **AGENT** | `agent.py` | `tools`, LangGraph |
| **TOOLS** | `tools.py` | `qa_chain`, `trafilatura` |
| **VISION** | `vision_extract.py` | Gemini API |
| **UI** | `app.py` | `agent`, `vision`, Gradio |

## File Structure
```text
Omni-RAG-Agent/
├── app.py                         # Root wrapper for Render/HF
├── requirements.txt
├── .gitignore
├── README.md
├── docs/                          # Architecture diagrams (.mmd, .puml, .txt)
├── data/                          # Raw PDFs and chunks.jsonl (gitignored)
├── index/                         # FAISS index and metadata (gitignored)
├── src/                           # Core source code
│   ├── app.py                     # Gradio UI
│   ├── agent.py                   # LangGraph ReAct agent
│   ├── tools.py                   # Search, Scrape, Shop, Expense
│   ├── vision_extract.py          # Gemini Vision OCR
│   ├── qa_chain.py                # Grounded Q&A with citations
│   ├── recipe1-4.py               # PDF -> FAISS pipeline
│   ├── config.py                  # Central paths and keys
│   └── schemas.py                 # Pydantic models
└── notebooks/                     # Smoke tests
```

## Phase Reference
I was built one phase at a time. Each phase has a specific architectural purpose.

| Phase | File | Purpose |
|---|---|---|
| 0 | `config.py` | Environment setup and Gemini API validation |
| 1 | `recipe1-3.py` | Ingestion, chunking, Gemini embeddings, FAISS index |
| 2 | `qa_chain.py` | Grounded QA, score gates, strict Pydantic citations |
| 3 | `agent.py`, `tools.py` | LangGraph ReAct agent, mock expense filing |
| 4 | `recipe4.py`, `tools.py` | Multi-PDF metadata filtering (oversampling) |
| 5 | `app.py` | Gradio UI, streaming status, subprocess handling |
| 6 | `vision_extract.py`, `tools.py` | Image OCR, URL scraping, mock shopping, SSRF guard |

## Signatures
| Function | Input | Output | Example |
|---|---|---|---|
| `retrieve(query, top_k)` | str, int | `list[dict]` with score, source, page | `retrieve("hotel budget")` |
| `answer_question(q, filter_source)` | str, str | `QAResult` (Pydantic model) | `answer_question("budget", "policy.pdf")` |
| `run_agent(query, verbose)` | str, bool | str (agent's final response) | `run_agent("Buy a pen for $2")` |
| `scrape_url(url)` | str | str (web text or error) | `scrape_url("https://example.com")` |
| `extract_text(image_path)` | str | str (OCR text) | `extract_text("receipt.png")` |
| `mock_add_to_cart(name, qty, price)` | str, int, float | str (cart status) | `mock_add_to_cart("Pen", 1, 2.0)` |
| `mock_place_order(address, confirmed)` | str, bool | str (order status) | `mock_place_order("123 Main", True)` |

## Formulas and Parameters
| Item | Definition |
|---|---|
| **Chunk Size** | 800 characters |
| **Chunk Overlap** | 150 characters |
| **Embedding Dim** | 768 (Gemini `gemini-embedding-001`) |
| **Cosine Similarity** | L2-normalized `IndexFlatIP` in FAISS |
| **Score Gate** | `max(score) < 0.3` triggers "I don't know" (no LLM call) |
| **Agent Recursion Limit** | 9 steps (supports up to 4 tool calls) |

## Rules I Follow
| Rule | Testable statement |
|---|---|
| **Score Gate** | If max similarity < 0.3, I return "not found" without calling Gemini. |
| **No Fake Citations** | Pydantic validators cross-check every source against retrieved chunks. |
| **Prompt Injection** | Retrieved text is wrapped in `<context>` tags and defined as DATA, not instructions. |
| **SSRF Protection** | `scrape_url` rejects `localhost`, `127.0.0.1`, `10.*`, `192.168.*`, etc. |
| **Mock Actions** | `mock_place_order` requires `confirmed=True` and a non-empty address. |
| **Chat Memory** | The last 4 turns (8 messages) are passed to the agent via `build_full_query()`. |
| **Image Safety** | Transparent images are flattened to white; corrupt files return `""` (no API call). |

## Testing
My tests are offline Python scripts and Jupyter notebooks. To run them:
```bash
python src/tools.py
python src/vision_extract.py --selftest
```

| Test | Coverage |
|---|---|
| `tools.py` | Expense validation, SSRF block list, cart merging, order confirmation. |
| `vision_extract.py` | Empty, corrupt, truncated, missing images (all return `""`). |
| `recipe4_retrieve.py` | Metadata filtering, case-insensitivity, "all" vs specific file. |

## Known Limitations
- I read text-based PDFs only. Scanned PDFs require the Phase 6 Image OCR tab.
- I do not scan subfolders; I read exactly where you point me.
- I do not retain your PDFs across restarts (unless deployed with persistent cloud storage).
- I serve one local session. I have no multi-user authentication.
- I do not process real payments or place real orders (mock tools only).

## Roadmap
Version 2 plans:
- **Multimodal RAG:** Feed images directly to the LLM instead of just OCR text.
- **Persistent Storage:** Swap FAISS for Pinecone to survive cloud restarts.
- **Real E-commerce:** Integrate Shopify or Stripe APIs for real checkouts.
- **Multi-User:** Add Supabase Auth to isolate users' indexes.

## Reference
This project is based on the "Actionable Insight Engine" roadmap (Phase 0-6), utilizing LangChain, LangGraph, and Google Gemini.