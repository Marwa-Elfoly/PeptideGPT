from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel, EsmForProteinFolding
import math
import torch
import numpy as np
import pandas as pd
import argparse
import yaml
import os
import shutil
import gc
from tqdm import tqdm

# === Directory Setup ===
def create_or_replace_directory(directory_path):
    if os.path.exists(directory_path):
        print(f"Directory {directory_path} exists. Deleting...")
        shutil.rmtree(directory_path)
    print(f"Creating directory {directory_path}...")
    os.makedirs(directory_path)

# === Argument Parser ===
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_return_sequences', type=int, default=1000)
parser.add_argument('--max_length', type=int, default=50)
parser.add_argument('--starts_with', type=str, default='')
parser.add_argument('--output_dir', type=str)
parser.add_argument('--pred_model_path', type=str)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_aa_count', type=int, default=25)  # ✅ Added

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# === Generation ===
create_or_replace_directory(args.output_dir)
protgpt2 = pipeline('text-generation', model=args.model_path, device_map="auto")
print('Starting generation...')
sequences = []
all_amines = ['L','A','G','V','E','S','I','K','R','D','T','P','N','Q','F','Y','M','H','C','W']
for cha in all_amines:
    generated = protgpt2(
        "<|endoftext|>" + cha,
        max_length=args.max_length,
        do_sample=True,
        top_k=950,
        repetition_penalty=1.2,
        num_return_sequences=args.num_return_sequences // len(all_amines),
        eos_token_id=0
    )
    sequences.extend(generated)
print(f"✅ Generated {len(sequences)} raw sequences.")

# ✅ Filter sequences by max AA count BEFORE scoring
filtered_sequences = []
for entry in sequences:
    seq = entry['generated_text'].replace("<|endoftext|>", "").replace('\n', '').strip()
    if len(seq) <= args.max_aa_count:
        filtered_sequences.append(seq)
print(f"✅ Retained {len(filtered_sequences)} sequences after filtering by max_aa_count = {args.max_aa_count}")

# === Perplexity Calculation ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)

def calculatePerplexity(sequence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs[0])

ppls = []
scored_sequences = []

print("⚙️ Calculating perplexity...")
for seq in tqdm(filtered_sequences):
    formatted = '\n'.join(seq[i:i+60] for i in range(0, len(seq), 60))
    seq_with_tags = f"<|endoftext|>{formatted}<|endoftext|>"
    ppl = calculatePerplexity(seq_with_tags, model, tokenizer)
    scored_sequences.append(seq)
    ppls.append(ppl)

# Save all results
df_ppl = pd.DataFrame({'Sequence': scored_sequences, 'Perplexity': ppls})
df_ppl.to_csv(os.path.join(args.output_dir, "all_generated_with_perplexity.csv"), index=False)

# === Select Top by Perplexity ===
k = args.num_return_sequences // 3
top_prots = np.array(scored_sequences)[np.argsort(ppls)[:k]]
top_prots = [i for i in top_prots if len(i) != 0]
print("✅ Selected top sequences by perplexity.")

# === Hull Validity Check ===
print("🔍 Checking hull validity...")
data = np.load('./hull_equations.npz')
def create_3d_point(sequence, c):
    for norm in 'XUBZOJ': sequence = sequence.replace(norm, '')
    count = sequence.count(c)
    if count == 0: return [0, 0, 0]
    indices = [i for i, char in enumerate(sequence) if char == c]
    mean = sum(indices) / count
    variance = sum((i - mean)**2 for i in indices) / count
    return [count, mean, variance / len(sequence)]

def point_in_hull(point, hull_equations, tol=1e-12):
    return all((np.dot(eq[:-1], point) + eq[-1] <= tol) for eq in hull_equations)

def check_validity(seq):
    return all(point_in_hull(create_3d_point(seq, aa), data[aa]) for aa in data.files)

df_valid = pd.DataFrame({'Sequence': top_prots})
df_valid['Valid Protein'] = df_valid['Sequence'].apply(check_validity)
filtered_prots = df_valid[df_valid['Valid Protein']]['Sequence'].tolist()
print("✅ Hull validity check complete.")

# === ESMFold Structure Check ===
print("🧠 Running ESMFold...")
esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

plddt_results = []
valid_for_structure = filtered_prots

for seq in tqdm(valid_for_structure):
    try:
        inputs = esm_tokenizer([seq.replace("\n", "")], return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = esm_model(**inputs)
        plddt_results.append(torch.mean(outputs.plddt).item())
    except Exception as e:
        print(f"⚠️ Skipping sequence due to error: {e}")
        plddt_results.append(None)

df_valid = df_valid[df_valid['Valid Protein']].copy()
df_valid['plDDT'] = plddt_results
df_valid['Good Structure'] = df_valid['plDDT'] > 0.7
df_valid.to_csv(os.path.join(args.output_dir, 'generation_checks.csv'), index=False)
best_prots = df_valid[df_valid['Good Structure']]['Sequence'].tolist()
print("✅ Structure check complete.")

# === PeptideBERT Property Prediction ===
from PeptideBERT.data.dataloader import load_data
from PeptideBERT.model.network import create_model, cri_opt_sch
from PeptideBERT.model.utils import train, validate, test

print("📊 Running PeptideBERT prediction...")
save_dir = args.pred_model_path
config = yaml.safe_load(open(os.path.join(save_dir, "config.yaml")))
config['device'] = device
peptideBERT_model = create_model(config)
peptideBERT_model.load_state_dict(torch.load(f"{save_dir}/model.pt")['model_state_dict'], strict=False)
peptideBERT_model.to(device)
peptideBERT_model.eval()

m2 = dict(zip(
    ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','L','A','G','V','E','S','I','K','R','D','T','P','N','Q','F','Y','M','H','C','W','X','U','B','Z','O'],
    range(30)
))

def encode(seq):
    seq_ids = [m2[a] for a in seq if a in m2]
    x = torch.tensor(seq_ids).unsqueeze(0).to(device)
    attn = torch.tensor(x > 0, dtype=torch.long).to(device)
    return x, attn

scores = {}
for seq in tqdm(best_prots):
    try:
        score = peptideBERT_model(*encode(seq.replace('\n', ''))).item()
        scores[seq] = score
    except:
        continue

df_preds = pd.DataFrame({
    'Sequence': list(scores.keys()),
    'Score': list(scores.values())
})
df_preds['Property'] = df_preds['Score'] > 0.5
df_preds.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

# === Summary Info ===
total = args.num_return_sequences
top = len(top_prots)
valid = len(filtered_prots)
good_struct = len(best_prots)
positive = sum(df_preds['Property'])

with open(os.path.join(args.output_dir, 'info.txt'), 'w') as f:
    f.write(f'Total generated sequences: {total}, top {top} used for scoring\n')
    f.write(f'{valid} passed hull validity check\n')
    f.write(f'{good_struct} had plDDT > 0.7\n')
    f.write(f'{positive}/{good_struct} had desired property ({positive/good_struct:.2%})\n')

print("✅ Full pipeline complete.")
