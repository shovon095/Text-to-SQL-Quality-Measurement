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
- Automatic **ablation study** (`full`, `no-feedback`, `no-semantic`) — visualized in `ablation.png`  
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
Example data format
json
Copy
Edit
{
  "db_id": "concert_singer",
  "question": "How many singers are from USA?",
  "SQL": "SELECT COUNT(*) FROM singer WHERE nation = 'US'",
  "evidence": "nation column stores the country",
  "difficulty": "simple"
}
SQLite file must be located at:
data/databases/<db_id>/<db_id>.sqlite

3. llama_quality_check.py — Open-weights pipeline
3.1 Run (LoRA fine-tuning or inference)
bash
Copy
Edit
python llama_quality_check.py \
  --do_train \
  --train_path   data/train.json \
  --eval_path    data/dev.json \
  --db_root_path data/databases \
  --engine       meta-llama/Llama-2-7b-hf \
  --output_dir   checkpoints/llama_ft \
  --batch_size   2 \
  --run_ablation
3.2 Key arguments
Flag	Default	Description
--do_train / --do_eval	None	Enables training or inference mode
--engine	LLaMA-2-7b	Any HF causal LM model
--lora_rank	16	LoRA rank used during PEFT
--norm_threshold	0.75	Similarity threshold to trigger semantic validation
--run_ablation	off	Enables plotting of performance breakdown (ablation.png)
--max_length	2048	Max input/output token length
--exec_timeout	20	SQLite timeout (in seconds) for semantic execution validation

3.3 Outputs
preds.json — Generated SQL per question

ablation.png — Mean similarity (bar chart) across pipeline variants

loft_logs/ — Attempt-by-attempt logs (per query)

checkpoints/llama_ft/ — Saved LoRA weights and tokenizer

4. updated_gpt_request.py — OpenAI API pipeline
4.1 Run (zero/few-shot with GPT-3.5/4/4o)
bash
Copy
Edit
export OPENAI_API_KEY="sk-..."

python updated_gpt_request.py \
  --eval_path     data/dev.json \
  --db_root_path  data/databases \
  --engine        gpt-4o-mini \
  --data_output_path preds/predict_dev.json \
  --use_knowledge True \
  --chain_of_thought True
4.2 Key arguments
Flag	Description
--mode	Choose dev or test split
--engine	Model name (e.g. gpt-4, gpt-4o-mini, gpt-3.5-turbo-instruct)
--use_knowledge	Appends evidence field to prompt
--chain_of_thought	Adds “Let’s think step-by-step.” + stores intermediate thoughts
--max_attempts	Retry attempts per example (default: 3)
--backoff_seconds	Start value for exponential backoff if rate-limited
--resultset_tolerance	Allows small row/value mismatches for semantic comparison

4.3 Outputs
predict_<split>(_cot).json — Best SQL (plus CoT if enabled)

feedback_results.txt — Detailed logs for all attempts

rate_limit.log — Retry logs in case of OpenAI 429/5xx

5. Choosing the Right Script
Use case	Script	Why
On-premises fine-tuning with full control	llama_quality_check.py	No external API, supports LoRA-based instruction tuning
Cloud-based inference with strong models	updated_gpt_request.py	Fast setup, great zero-shot performance with GPT-4 / GPT-4o

6. Troubleshooting
CUDA/torch device mismatch → Patched inside LLaMA script

OpenAI rate limits → Auto-retries with exponential backoff

Long prompt or OOM → Try lowering --max_length, remove --use_knowledge

Database not found → Must follow: data/databases/<db_id>/<db_id>.sqlite

7. Contributing
Fork ➜ new branch ➜ PR

Run pre-commit run --all-files (runs black, isort, etc.)

Want a new backend? Add subcommand via the wrapper in quality_check.py

📄 License
Released under the MIT License.
See LICENSE for full terms.

✏️ Citation
pgsql
Copy
Edit
S. Sarker et al., “Enhancing LLM Fine-Tuning for Text-to-SQL by SQL Quality
Measurement,” manuscript under review, 2025.
Happy querying! 🎉

vbnet
Copy
Edit

This version is copy-paste ready into your single `README.md` file. All headers will compile, navigation will work in GitHub, and nothing is broken or fragmented. Let me know if you'd like badges (e.g. model, license, Python version) or rendered screenshots added.







