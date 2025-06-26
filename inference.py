# inference.py
from transformers import pipeline
import math
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, EsmForProteinFolding
import numpy as np
import pandas as pd
import argparse
import yaml
import os
import shutil
import gc
from tqdm import tqdm

def create_or_replace_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

# ── Arguments ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model_path',          type=str, required=True)
parser.add_argument('--num_return_sequences', type=int, default=1000)
parser.add_argument('--max_length',          type=int, default=50)
parser.add_argument('--starts_with',         type=str, default='')
parser.add_argument('--output_dir',          type=str, required=True)
parser.add_argument('--pred_model_path',     type=str, required=True)
parser.add_argument('--seed',                type=int, default=42)
parser.add_argument('--max_aa_count',        type=int, default=25, help="(for annotation only)")
args = parser.parse_args()

# ── Seeding ─────────────────────────────────────────────────────────────────────
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── Setup ──────────────────────────────────────────────────────────────────────
create_or_replace_directory(args.output_dir)
protgpt2 = pipeline('text-generation', model=args.model_path, device_map="auto")

# ── Generation ─────────────────────────────────────────────────────────────────
print('🔄 Generating sequences...')
sequences = []
all_amines = list("LAGVESIKRDTPNQFYMHCW")
for cha in all_amines:
    batch = protgpt2(
        "<|endoftext|>" + cha,
        max_length=args.max_length,
        do_sample=True,
        top_k=950,
        repetition_penalty=1.2,
        num_return_sequences=args.num_return_sequences // len(all_amines),
        eos_token_id=0
    )
    sequences.extend(batch)
print(f"✅ Generated {len(sequences)} raw sequences.")

# ── Perplexity Scoring ─────────────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
model     = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)

def calculatePerplexity(sequence):
    ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0).to(device)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return math.exp(loss)

print('⚙️ Calculating perplexity...')
ppls = []
seqs = []
for entry in tqdm(sequences):
    s = entry['generated_text'].replace("<|endoftext|>", "").strip().replace("\n","")
    formatted = '<|endoftext|>' + '\n'.join(s[i:i+60] for i in range(0,len(s),60)) + '<|endoftext|>'
    seqs.append(s)
    ppls.append(calculatePerplexity(formatted))

# ── Write Perplexity CSV (with AA_Count) ────────────────────────────────────────
df = pd.DataFrame({'Sequence': seqs, 'Perplexity': ppls})
df['AA_Count'] = df['Sequence'].str.len()
out1 = os.path.join(args.output_dir, 'all_generated_with_perplexity.csv')
df.to_csv(out1, index=False)
print(f"✅ Saved perplexity + AA_Count: {out1}")

# ── Top-K Selection ─────────────────────────────────────────────────────────────
k = args.num_return_sequences // 3
top_prots = df.sort_values('Perplexity').head(k)['Sequence'].tolist()
print(f"✅ Selected top {len(top_prots)} sequences by perplexity.")

# ── Hull Filtering ──────────────────────────────────────────────────────────────
data = np.load('./hull_equations.npz')
def create_3d_point(seq, c):
    for x in "XUBZOJ": seq = seq.replace(x,'')
    cnt = seq.count(c)
    if cnt == 0: return [0,0,0]
    idxs = [i for i, ch in enumerate(seq) if ch == c]
    mean = sum(idxs) / cnt
    var  = sum((i - mean)**2 for i in idxs) / cnt
    return [cnt, mean, var/len(seq)]

def in_hull(pt, eqs, tol=1e-12):
    return all(np.dot(e[:-1], pt) + e[-1] <= tol for e in eqs)

def is_valid(seq):
    return all(in_hull(create_3d_point(seq, c), data[c]) for c in data.files)

df_val = pd.DataFrame({'Sequence': top_prots})
df_val['Valid'] = df_val['Sequence'].apply(is_valid)
valids = df_val[df_val['Valid']]['Sequence'].tolist()
print(f"✅ {len(valids)} passed hull check.")

# ── ESMFold Structure ──────────────────────────────────────────────────────────
esm_tok   = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
esm_model = EsmForProteinFolding.from_pretrained(
