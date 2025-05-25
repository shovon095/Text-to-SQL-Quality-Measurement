# Text-to-SQL Quality-Check 📊

> **Two complementary pipelines** for converting natural-language questions to SQL, grading the output (string **and** semantic similarity), and self-refining the query until a quality threshold is reached.

| Script                    | Model family                           | Typical scenario                                       |
|---------------------------|----------------------------------------|--------------------------------------------------------|
| `llama_quality_check.py`  | **Open-weights LLaMA-2 / HF causal-LM**| On-prem GPU fine-tuning with LoRA + 8-bit quantization |
| `updated_gpt_request.py`  | **OpenAI GPT-3.5 / GPT-4 / GPT-4o**    | Rapid prototyping via OpenAI API (no GPU required)     |

---

## ✨ Core Features

- Difficulty-aware prompts (simple / moderate / complex)  
- Dual grading – **string similarity** *and* **semantic (result-set) equivalence**  
- Up to **three** self-refinement attempts with structured feedback  
- Optional **ablation study** (`full`, `no-feedback`, `no-semantic`) → `ablation.png`  
- Clean JSON outputs, feedback logs, and optional Chain-of-Thought (CoT) tracking  

---

## 1. Installation

```bash
# Common dependencies (Python 3.9+ recommended)
pip install pandas tqdm sqlparse rapidfuzz matplotlib

# For LLaMA-based fine-tuning
pip install torch transformers peft bitsandbytes

# For OpenAI API
pip install openai backoff
2. Dataset & Folder Structure
php-template
Copy
Edit
repo/
├── data/
│   ├── train.json
│   ├── dev.json
│   └── databases/
│       └── <db_id>/<db_id>.sqlite
├── llama_quality_check.py
└── updated_gpt_request.py
