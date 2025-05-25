# Text-to-SQL Quality-Check 📊

> **Two complementary pipelines** for converting natural-language
> questions to SQL, grading the output (string **and** semantic similarity),
> and self-refining the query until a quality threshold is reached.

| Script                              | Model family                         | Typical scenario                                       |
|-------------------------------------|--------------------------------------|--------------------------------------------------------|
| `llama_quality_check.py`            | **Open-weights LLama-2 / HF causal-LM** | On-prem GPU fine-tuning with LoRA + 8-bit quantisation |
| `updated_gpt_request.py`            | **OpenAI GPT-3.5 / GPT-4 / GPT-4o**  | Rapid prototyping via OpenAI API (no GPU required)     |

---

## ✨ Core Features (both scripts)

* Difficulty-aware prompts (simple / moderate / complex)  
* Dual grading – **string similarity** *and* **semantic (result-set) equivalence*  
* Up to **three** self-refinement attempts with structured feedback  
* Automatic **ablation study** (`full`, `no-feedback`, `no-semantic`) – bar chart saved as `ablation.png`  
* Clean JSON outputs, feedback logs, and optional COT retention  

---

## 1  Installation

```bash
# Common utilities (Python 3.9+ recommended)
python -m pip install pandas tqdm sqlparse rapidfuzz matplotlib

# ——— Local Llama workflow ———
python -m pip install "torch>=2.1" "transformers>=4.52.3" peft bitsandbytes

# ——— OpenAI workflow ———
python -m pip install openai backoff
2 Dataset & Folder Layout
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
JSON record (example)

jsonc
Copy
Edit
{
  "db_id": "concert_singer",
  "question": "How many singers are from USA?",
  "SQL": "SELECT COUNT(*) FROM singer WHERE nation = 'US'",
  "evidence": "nation column stores the country",
  "difficulty": "simple"
}
3 llama_quality_check.py — Open-weights pipeline
3.1 Command-line quick-start
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
Flag	Default	Meaning
--do_train / --do_eval	None	Enable training or inference only
--engine	Llama-2-7b	Any HF causal-LM checkpoint
--lora_rank	16	LoRA rank for fine-tuning
--norm_threshold	0.75	Min. similarity (0–1) that triggers semantic checks
--run_ablation	off	Save ablation.png comparing pipeline variants
--max_length	2048	Truncate prompt + generation to this many tokens
--exec_timeout	20 s	Seconds before SQLite execution is killed

3.3 Outputs
preds.json – generated SQL per question

ablation.png – bar chart of mean similarity for three variants

loft_logs/ – structured logs for each refinement attempt

checkpoints/llama_ft/ – PEFT model & tokenizer (if --do_train)

4 updated_gpt_request.py — OpenAI pipeline
4.1 Command-line quick-start
bash
Copy
Edit
export OPENAI_API_KEY="sk-···"

python updated_gpt_request.py \
  --eval_path     data/dev.json \
  --db_root_path  data/databases \
  --engine        gpt-4o-mini \
  --data_output_path preds/predict_dev.json \
  --use_knowledge True \
  --chain_of_thought True
4.2 Important arguments
Flag	Purpose
--mode (dev / test)	Select split processed
--engine	Any Chat/Completion model (e.g. gpt-4o-mini)
--use_knowledge	Append evidence to the prompt
--chain_of_thought	Add “Let’s think step-by-step” and keep COT in JSON
--max_attempts	Number of refinement rounds (default = 3)
--backoff_seconds	Start value for exponential back-off on rate-limit
--resultset_tolerance	Allowed row-order / float diff for semantic match

4.3 Outputs
predict_<split>([_cot]).json – best SQL per question (+ COT if kept)

feedback_results.txt – human-readable trace of each attempt

rate_limit.log – timestamps of any 429 or 5xx retries

5 Choosing the right script
You need…	Use this script	Why?
Keep data on-prem, fine-tune open model	llama_quality_check.py	LoRA + 8-bit, no external API calls
Rapid results, no GPU, frontier model quality	updated_gpt_request.py	Minimal dependencies, GPT-4(/4o) accuracy

6 Troubleshooting
CUDA / torch.bmm device mix – the Llama script monkey-patches common HF quirks.

OpenAI rate limits – automatic exponential back-off; tune --backoff_seconds.

Long prompts / OOM – lower --max_length or omit --use_knowledge.

Incorrect DB path – check that data/databases/<db_id>/<db_id>.sqlite exists.

7 Contributing
Fork ➜ new branch ➜ PR.

Run pre-commit run --all-files (black + isort).

For new models, add a sub-parser entry in the wrapper quality_check.py.

📄 License
Released under the MIT License.
See LICENSE for details.

✏️ Citation
pgsql
Copy
Edit
S. Sarker et al., “Enhancing LLM Fine-Tuning for Text-to-SQL by SQL Quality
Measurement,” manuscript under review, 2025.
Happy querying 🎉

Copy
Edit
