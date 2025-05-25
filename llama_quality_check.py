#!/usr/bin/env python3
# --------------------------------------------------------------------------
# Monkey-patches for torchvision / HF 4.52.3 issues and device mismatches
# --------------------------------------------------------------------------
# ------------------ bitsandbytes + DTensor patch ------------------
# ------------------ bitsandbytes + DTensor patch ------------------
# ------------------ bitsandbytes + DTensor workaround ------------------
# ------------------ bitsandbytes + DTensor fallback ------------------
# ------------------ bitsandbytes + DTensor fallback ------------------
import sys, types

tp_stub = types.ModuleType("transformers.integrations.tensor_parallel")

# Comprehensive stubs for all current expected methods
tp_stub.ALL_PARALLEL_STYLES = ["colwise", "rowwise", "megatron_lm", "tp"]
tp_stub._get_parameter_tp_plan = lambda *args, **kwargs: None
tp_stub._prepare_output_fn = lambda *args, **kwargs: lambda *a, **kw: None
tp_stub._get_tensor_parallel_model = lambda *args, **kwargs: None
tp_stub.get_tensor_model_parallel_group = lambda: None
tp_stub.model_is_tp = lambda x: False
tp_stub.prepare_model_for_tp = lambda x, *a, **kw: x
tp_stub.save_model_with_tp = lambda *a, **kw: None
tp_stub.initialize_tensor_parallelism = lambda *a, **kw: (None, None, None)
tp_stub.repack_weights = lambda model: model
tp_stub.replace_state_dict_local_with_dtensor = lambda state_dict, *args, **kwargs: state_dict

# Add missing method causing your latest error:
tp_stub.shard_and_distribute_module = lambda module, *args, **kwargs: module

# Add the missing verify_tp_plan function that's causing the import error
tp_stub.verify_tp_plan = lambda *args, **kwargs: True

# Inject fake module into sys.modules
sys.modules["transformers.integrations.tensor_parallel"] = tp_stub



import torch

# 1) patch torch.library.register_fake for torchvision compatibility
if not hasattr(torch, "library"):
    torch.library = types.SimpleNamespace(
        register_fake=lambda *args, **kwargs: (lambda fn: fn)
    )
elif not hasattr(torch.library, "register_fake"):
    torch.library.register_fake = lambda *args, **kwargs: (lambda fn: fn)

# 2) patch torch.get_default_device for Transformers 4.52.3
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

# 3) patch torch.Tensor.__matmul__ to auto-cast the RHS to self.device
_orig_matmul = torch.Tensor.__matmul__
def _patched_matmul(self, other):
    if isinstance(other, torch.Tensor) and self.device != other.device:
        other = other.to(self.device)
    return _orig_matmul(self, other)
torch.Tensor.__matmul__ = _patched_matmul

# 4) patch torch.bmm for explicit batched‐matmul mismatches
_orig_bmm = torch.bmm
def _patched_bmm(a, b):
    if isinstance(b, torch.Tensor) and a.device != b.device:
        b = b.to(a.device)
    return _orig_bmm(a, b)
torch.bmm = _patched_bmm

# --------------------------------------------------------------------------
# All other imports
# --------------------------------------------------------------------------
import argparse, json, os, sqlite3, csv
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

def find_sqlite_file(db_root: str, db_id: str):
    cand = Path(db_root) / db_id / f"{db_id}.sqlite"
    if cand.is_file(): return str(cand)
    folder = Path(db_root) / db_id
    if folder.is_dir():
        for f in folder.iterdir():
            if f.suffix.lower() in ('.sqlite','.db','.sqlite3'):
                return str(f)
    for ext in ('.sqlite','.db','.sqlite3'):
        flat = Path(db_root) / f"{db_id}{ext}"
        if flat.is_file(): return str(flat)
    return None

def get_schema_and_examples(db_path: str, num_rows: int=3):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    schema_lines, example_lines = [], []
    for tbl, ddl in cur.fetchall():
        if not ddl: continue
        schema_lines.append(ddl.strip() + ';')
        if num_rows > 0:
            try:
                cur.execute(f"PRAGMA table_info('{tbl}')")
                cols = [r[1] for r in cur.fetchall()]
                cur.execute(f"SELECT * FROM '{tbl}' LIMIT {num_rows}")
                rows = cur.fetchall()
                if cols and rows:
                    example_lines.append(f"-- Table: {tbl}")
                    example_lines.append(' | '.join(cols))
                    for r in rows:
                        example_lines.append(' | '.join(map(str,r)))
                    example_lines.append('')
            except:
                pass
    return '\n'.join(schema_lines), '\n'.join(example_lines)

def normalize_sql(sql: str) -> str:
    return ' '.join(sql.lower().strip().split())

def calculate_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() * 100

def semantic_comparison(gen_sql: str, truth_sql: str, db_path: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute(gen_sql); pred = cur.fetchall()
        cur.execute(truth_sql); true = cur.fetchall()
        return Counter(pred) == Counter(true)
    except:
        return False

def compute_difficulty(sim: float, num_cols: int, threshold: float) -> str:
    if sim >= threshold and num_cols <= 5:   return 'simple'
    if sim <  threshold and num_cols <  10:  return 'moderate'
    return 'complex'

def refine_prompt(base: str, diff: str) -> str:
    hints = {
      'simple':   "Focus on selecting the right columns and applying basic filters.",
      'moderate': "Ensure proper JOINs and accurate WHERE clauses.",
      'complex':  "Step-by-step: analyze schema, plan joins, outline filters & aggregation, then construct SQL."
    }
    return f"{base}\n-- Difficulty Hint: {hints[diff]}"

def prepare_base_prompt(q: str, sch: str, ex: str) -> str:
    parts = []
    if sch: parts.append("SQLite schema:\n" + sch)
    if ex: parts.append("Example rows:\n" + ex)
    parts.append(f"Question: {q}")
    parts.append("SQL Query:")
    return '\n\n'.join(parts) + ' '

def load_data(path: str) -> list:
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        elif isinstance(data, list):
            return data
        else:
            return [data]
    elif path.endswith('.jsonl'):
        return [json.loads(line) for line in open(path, 'r', encoding='utf-8') if line.strip()]
    elif path.endswith('.csv'):
        return list(csv.DictReader(open(path, 'r', encoding='utf-8')))
    else:
        return []

# --------------------------------------------------------------------------
# HF / PEFT setup
# --------------------------------------------------------------------------
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# --------------------------------------------------------------------------
# Training with Feedback & Ablation
# --------------------------------------------------------------------------
def train_with_feedback(args):
    tnc = args.norm_threshold
    tokenizer = AutoTokenizer.from_pretrained(
        args.engine, token=args.hf_token, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ──────────── load 8-bit model ────────────
    base_model = AutoModelForCausalLM.from_pretrained(
        args.engine,
        load_in_8bit=True,
        device_map="auto",
        token=args.hf_token
    )

    # ──── ensure all RoPE inv_freq on CUDA ────
    device = torch.get_default_device()
    for module in base_model.modules():
        if hasattr(module, "inv_freq"):
            module.inv_freq = module.inv_freq.to(device)

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=['q_proj','k_proj','v_proj','o_proj'],
        task_type=TaskType.CAUSAL_LM
    )

    variants = {
      'full':        {},
      'no_feedback': {'compute_difficulty': lambda sim,n,th: 'simple'},
      'no_semantic': {'semantic_comparison': lambda g,t,db: False}
    }

    train_examples = load_data(args.train_path)
    results = {}

    for name, patch in variants.items():
        globals().update(patch)
        model = get_peft_model(base_model, lora_cfg).train()
        torch.manual_seed(args.seed)

        processed = []
        for ex in train_examples:
            dbid     = ex.get('db_id')
            question = ex.get('question')
            gt_sql   = ex.get('SQL') or ex.get('sql')
            if not (dbid and question and gt_sql): continue

            dbp = find_sqlite_file(args.db_root_path, dbid)
            if not dbp: continue

            sch, exs = get_schema_and_examples(dbp)
            base_prompt = prepare_base_prompt(question, sch, exs)

            best_score, best_sql = -1.0, gt_sql
            prompt = base_prompt

            for attempt in range(1,4):
                inp = tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    padding='max_length',
                    max_length=args.max_length
                ).to(device)

                out = model.generate(**inp, max_new_tokens=256)
                gen_sql = tokenizer.decode(
                    out[0][ inp['input_ids'].shape[1] : ],
                    skip_special_tokens=True
                ).strip()

                norm_gen = normalize_sql(gen_sql)
                norm_gt  = normalize_sql(gt_sql)
                sim      = calculate_similarity(norm_gen, norm_gt)

                sem_ok = False
                if sim >= tnc:
                    sem_ok = semantic_comparison(gen_sql, gt_sql, dbp)

                if sem_ok or sim > best_score:
                    best_score, best_sql = sim, gen_sql
                if sem_ok:
                    break

                print(f"[Train][{name}] DB:{dbid} Att{attempt}: sim={sim:.2f}%, sem_ok={sem_ok}")

                ncols = len(gen_sql.split())
                diff  = compute_difficulty(sim, ncols, tnc)
                prompt = refine_prompt(base_prompt, diff)

            processed.append((prompt + best_sql, best_sql))

        # build fine-tune dataset
        X, Y = [], []
        for inp, out in processed:
            tf  = tokenizer(inp, truncation=True, padding='max_length', max_length=args.max_length)
            ids = tf['input_ids']
            tl  = tokenizer(inp+out, truncation=True, padding='max_length', max_length=args.max_length)
            cut = sum(1 for x in tokenizer(inp)['input_ids'] if x != tokenizer.pad_token_id)
            lbl = [-100]*cut + tl['input_ids'][cut:]
            lbl = lbl[:len(ids)] + [-100]*(len(ids)-len(lbl))
            X.append(ids); Y.append(lbl)

        class DS(torch.utils.data.Dataset):
            def __init__(self,x,y): self.x, self.y = x,y
            def __len__(self):    return len(self.x)
            def __getitem__(self,i):
                ids = torch.tensor(self.x[i]); lab = torch.tensor(self.y[i])
                mask= (ids!=tokenizer.pad_token_id).long()
                return {"input_ids":ids, "attention_mask":mask, "labels":lab}

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=os.path.join(args.output_dir,name),
                num_train_epochs=args.num_train_epochs,
                per_device_train_batch_size=args.per_device_train_batch_size,
                learning_rate=args.learning_rate,
                fp16=torch.cuda.is_available(),
                evaluation_strategy='steps',
                eval_steps=200,
                logging_steps=50,
                save_steps=200,
                load_best_model_at_end=True,
                save_total_limit=1,
                remove_unused_columns=False,
            ),
            train_dataset=DS(X,Y),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            tokenizer=tokenizer,
            compute_metrics=lambda p: {
              "sim": calculate_similarity(
                  normalize_sql(tokenizer.decode(p.predictions[0])),
                  normalize_sql(p.label_ids[0])
              )
            }
        )
        trainer.train()

        # evaluate for ablation
        dev = load_data(args.eval_path)
        sims = []
        for idx, ex in enumerate(dev,1):
            dbp = find_sqlite_file(args.db_root_path, ex['db_id'])
            sch, exs = get_schema_and_examples(dbp)
            prompt = prepare_base_prompt(ex['question'], sch, exs)
            inp    = tokenizer(
                prompt, return_tensors='pt',
                truncation=True, padding='max_length',
                max_length=args.max_length
            ).to(device)
            out    = model.generate(**inp, max_new_tokens=256)
            pred   = tokenizer.decode(
                       out[0][ inp['input_ids'].shape[1]: ],
                       skip_special_tokens=True
                    ).strip()
            sim    = calculate_similarity(
                       normalize_sql(pred),
                       normalize_sql(ex.get('SQL') or ex.get('sql'))
                    )
            sims.append(sim)
            print(f"[Eval][{name}] {idx}/{len(dev)}: sim={sim:.2f}%")

        results[name] = sum(sims)/len(sims) if sims else 0

    if args.run_ablation:
        plt.bar(results.keys(), results.values())
        plt.ylabel('Avg Normalized Similarity (%)')
        plt.title('Ablation Results')
        os.makedirs(args.output_dir, exist_ok=True)
        plt.savefig(os.path.join(args.output_dir,'ablation.png'))
        print('Ablation study scores:', results)

    print("\n=== Novelty Suggestions ===")
    print("1) Gate semantic checks behind a normalized threshold to reduce DB calls.")
    print("2) Dynamically adjust normalized threshold per-sample based on initial similarity.")
    print("3) Incorporate structural SQL embeddings for richer difficulty estimation.")

    return base_model

# --------------------------------------------------------------------------
# Inference (single-pass)
# --------------------------------------------------------------------------
def inference(args, model):
    tokenizer = AutoTokenizer.from_pretrained(
        args.engine, token=args.hf_token, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    # load 8-bit eval model
    model = AutoModelForCausalLM.from_pretrained(
        args.engine,
        load_in_8bit=True,
        device_map="auto",
        token=args.hf_token
    )

    # move RoPE inv_freq to CUDA
    device = torch.get_default_device()
    for module in model.modules():
        if hasattr(module, "inv_freq"):
            module.inv_freq = module.inv_freq.to(device)

    preds = []
    dev = load_data(args.eval_path)
    for idx, ex in enumerate(dev,1):
        dbp = find_sqlite_file(args.db_root_path, ex['db_id'])
        sch, exs = get_schema_and_examples(dbp)
        prompt = prepare_base_prompt(ex['question'], sch, exs)
        inp    = tokenizer(
            prompt, return_tensors='pt',
            truncation=True, padding='max_length',
            max_length=args.max_length
        ).to(device)
        out    = model.generate(**inp, max_new_tokens=256)
        sql    = tokenizer.decode(
                    out[0][ inp['input_ids'].shape[1]: ],
                    skip_special_tokens=True
                 ).strip()
        preds.append(f"{sql}\t----- bird -----\t{ex['db_id']}")
        print(f"[Infer] {idx}/{len(dev)} generated for {ex['db_id']}")

    os.makedirs(os.path.dirname(args.data_output_path), exist_ok=True)
    with open(args.data_output_path,'w') as f:
        json.dump(preds, f, indent=2)
    print('Predictions saved to', args.data_output_path)

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_path')
    p.add_argument('--eval_path')
    p.add_argument('--db_root_path',  required=True)
    p.add_argument('--engine',        default='meta-llama/Llama-2-7b-hf')
    p.add_argument('--hf_token',      help='HF token for gated models')
    p.add_argument('--output_dir',    default='./llama_ft')
    p.add_argument('--data_output_path', default='./preds.json')
    p.add_argument('--do_train',      action='store_true')
    p.add_argument('--do_eval',       action='store_true')
    p.add_argument('--num_train_epochs',            type=int,   default=3)
    p.add_argument('--per_device_train_batch_size', type=int,   default=4)
    p.add_argument('--learning_rate',               type=float, default=3e-4)
    p.add_argument('--max_length',                  type=int,   default=2048)
    p.add_argument('--seed',                        type=int,   default=42)
    p.add_argument('--run_ablation',                action='store_true')
    p.add_argument('--norm_threshold',              type=float, default=80.0,
                   help='Norm-sim threshold to gate semantic checks')
    args = p.parse_args()

    model = None
    if args.do_train:
        model = train_with_feedback(args)
    if args.do_eval:
        inference(args, model)