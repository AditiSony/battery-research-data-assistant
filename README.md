# battery-research-data-assistant 🔋🤖

Hello!
Welcome to my learning project on building a "Research Assistant" tool, an intelligent RAG (Retrieval-Augmented Generation) pipeline designed to query available scientific documents on battery technology with source traceability.

## 🏗 System Architecture
This project uses a "History-Aware Retrieval" logic to ensure conversational continuity.
```mermaid
graph TD
    A[User Query] --> B{Chat History?}
    B -- Yes --> C[Query Re-writer / LLM]
    C --> D[Enhanced Query]
    B -- No --> D
    D --> E[ChromaDB Vector Store]
    E --> F[Top-K Document Chunks]
    F --> G[Custom Formatter: Assign Source IDs]
    G --> H[Final Prompt: Context + Question]
    H --> I[Llama 3.2 via Ollama]
    I --> J[Answer with Inline Citations]
  ```

## ✨ Key Features
- Conversational Memory: Uses LangChain LCEL to re-contextualize follow-up questions (e.g., "What about its density?").

- Hardware Optimization: Configured to run Llama 3.2 3B locally, optimized for GPUs with ~4GB VRAM (like the NVIDIA T500).

- Source Citations: Implements a custom formatter to provide professional inline citations (e.g., [Source 1]) mapped to specific source document.

- Local-First: All embeddings and LLM processing stay on local machine using Ollama and HuggingFace Transformers.

## 🛠 Tech Stack
- Language: Python 3.11+
- Orchestration: LangChain (LCEL)
- LLM: Llama 3.2 (via Ollama)
- Embeddings: `BAAI/bge-small-en-v1.5`
- Vector Database: ChromaDB

## 📊 Data Source
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/batterydata/paper-abstracts)
This project uses the **paper-abstracts** dataset provided by **batterydata** from Hugging Face datasets. It contains abstracts of research papers related to battery technology as well as many non-battery-related papers.
- **Filtering:** Only abstracts labeled "battery" are ingested.
- **Augmentation:** > 💡 **Note:** A fictitious `journal` column is programmatically added during ingestion to demonstrate the RAG pipeline's ability to cite specific sources.

**Citation**
@article{huang2022batterybert,
  title={BatteryBERT: A Pretrained Language Model for Battery Database Enhancement},
  author={Huang, Shu and Cole, Jacqueline M},
  journal={J. Chem. Inf. Model.},
  year={2022},
  doi={10.1021/acs.jcim.2c00035},
  url={DOI:10.1021/acs.jcim.2c00035},
  pages={DOI: 10.1021/acs.jcim.2c00035},
  publisher={ACS Publications}
}

## 🚀 Getting Started
1. Prerequisites
- Install Ollama.
- Download the model:

Bash
ollama pull llama3.2

2. Installation
- Clone the repository and install dependencies using Pipenv:

Bash
pipenv install
pipenv shell

3. Usage
- Check config.yaml file to ensure all configurations are set as required.
- Ingest Data: Create the vector database by processing the abstracts.

Bash
python src/data.py

-Start Assistant: Launch the interactive chat loop.

Bash
python src/rag.py
