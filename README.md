# Agentic Document Intelligence & Industrialized Cross-Reference Engine

A production-grade multi-agent system leveraging the Model Context Protocol (MCP) to connect Frontier LLMs with legacy engineering repositories, automating complex cross-referencing tasks that previously required extensive manual human review.

---

## 🚀 Project Overview

This project demonstrates an end-to-end Agentic Document Intelligence pipeline built to automate engineering document cross-referencing using MCP, ReAct-based Multi-Agent Patterns, Multimodal RAG with Vision-Language Models, and an LLM-as-a-Judge (G-Eval) validation framework — achieving a 15% increase in data accuracy while eliminating manual audit bottlenecks.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `1_mcp_server_setup.py` | MCP server exposing PDF blueprints & Excel repositories as standardized queryable resources |
| `2_react_agent.py` | ReAct-based Sequential Multi-Agent Pattern with dedicated Extractor & Verifier agents |
| `3_multimodal_rag.py` | Multimodal RAG pipeline using Vision-Language Models (VLMs) & semantic chunking with ChromaDB |
| `4_llm_judge_eval.py` | LLM-as-a-Judge (G-Eval) framework for deterministic validation of agent outputs |
| `5_inference_optimization.py` | KV Caching + temperature/top-p tuning for latency reduction and controlled generation |
| `requirements.txt` | All dependencies required to run this project |

---

## 🛠️ Tech Stack

- **Agentic Framework:** Model Context Protocol (MCP), ReAct (Thought-Action-Observation), Sequential Multi-Agent Patterns
- **LLM:** Claude (Anthropic) — claude-sonnet-4-20250514
- **RAG Pipeline:** Multimodal RAG, Vision-Language Models (VLMs), LangChain, ChromaDB, Semantic Chunking
- **Evaluation:** LLM-as-a-Judge (G-Eval), Deterministic Validation
- **Inference Optimization:** KV Caching (Prompt Caching), Temperature / Top-k / Top-p Tuning
- **Document Processing:** PyMuPDF (fitz), OpenPyXL
- **Access Control:** MCP Roots for Sandboxed Resource Access

---

## 📊 Key Results

- ✅ Achieved a **15% increase in data accuracy** through deterministic G-Eval validation
- ✅ Fully eliminated manual audit bottlenecks across high-volume engineering schematics
- ✅ Standardized access to legacy PDF blueprints and Excel repositories via MCP interface
- ✅ Reduced inference latency through KV Caching and optimized decoding configurations
- ✅ Preserved spatial relationships in technical drawings using Vision-Language Model analysis

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/gayathrikumari/Agentic-Document-Intelligence.git

# Navigate to the project directory
cd Agentic-Document-Intelligence

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_api_key_here
```

---

## 📂 Data Setup

Create the following folder structure before running:

```
data/
├── blueprints/        ← Place your PDF engineering blueprints here
│   ├── blueprint1.pdf
│   └── blueprint2.pdf
└── master_book.xlsx   ← Place your Excel master book here
```

---

## 🔢 Run Order

Execute the scripts in the following order for the full pipeline:

```bash
# Step 1: Start the MCP server
python 1_mcp_server_setup.py

# Step 2: Run the ReAct Multi-Agent pipeline
python 2_react_agent.py

# Step 3: Ingest documents and test RAG queries
python 3_multimodal_rag.py

# Step 4: Validate agent outputs with G-Eval
python 4_llm_judge_eval.py

# Step 5: Benchmark inference configurations
python 5_inference_optimization.py
```

---

## 🏗️ Architecture

```
PDF Blueprints + Excel Master Book
        ↓
  MCP Server (1_mcp_server_setup.py)
  [Standardized Resource Interface]
        ↓
  ReAct Multi-Agent Pipeline (2_react_agent.py)
  [Extractor Agent → Verifier Agent]
        ↓
  Multimodal RAG (3_multimodal_rag.py)
  [VLM Image Analysis + Semantic Chunking + ChromaDB]
        ↓
  G-Eval Validation (4_llm_judge_eval.py)
  [Deterministic LLM-as-a-Judge]
        ↓
  Inference Optimization (5_inference_optimization.py)
  [KV Caching + Temperature Tuning]
```

---

## 👩‍💻 Author

**Gayathri Kumar**  
Data Scientist & AI Engineer | Ex-Deloitte | MS Information Science @ UNT  
📧 gayathrikumar.tx@gmail.com  
🔗 [([Linkedin](https://www.linkedin.com/in/gayathri-kumar-link/))](#) | [GitHub](#)

---

