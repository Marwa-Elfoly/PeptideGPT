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
        print(f"Directory {directory_path} exists. Deleting...")
        shutil.rmtree(directory_path)
        print(f"Creating directory {directory_path}...")
        os.makedirs(directory_path)
    else:
        print(f"Directory {directory_path} does not exist. Creating...")
        os.makedirs(directory_path)

# === Parser
parser = argparse.ArgumentParser(description='Protein sequence generation and scoring')
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

create_or_replace_directory(args.output_dir)
protgpt2 = pipeline('text-generation', model=args.model_path, device_map="auto")

print('Starting generation...')
sequences = []
all_amines = ['L','A','G','V','E','S','I','K','R','D','T','P','N','Q','F','Y','M','H','C','W']
for cha in all_amines:
    sequences_cha = protgpt2("<|endoftext|>" + cha, max_length=args.max_length, do_sample=True, top_k=950, repetition_penalty=1.2, num_return_sequences=args.num_return_sequences//len(all_amines), eos_token_id=0)
    sequences.extend(sequences_cha)
print('Generation Complete.')

# ✅ Added: Filter by max_aa_count before calculating perplexity
sequences = [
    seq for seq in sequences 
    if len(seq['generated_text'].replace("<|endoftext|>", "").strip()) <= args.max_aa_count
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using Device', device)

tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
model = GPT2LMHeadModel.from_pretrained(args.model_path)
model = model.to(device)

def calculatePerplexity(sequence, model, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sequence)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return math.exp(loss)

print('Starting to order by perplexity...')
ppls, sequences_with_ppl = [], []

for sequence in tqdm(sequences):
    seq = sequence['generated_text'].replace("<|endoftext|>", "").replace('\n', '').strip()
    sequences_with_ppl.append(seq)
    formatted = '<|endoftext|>' + '\n'.join(seq[i:i+60] for i in range(0, len(seq), 60)) + '<|endoftext|>'
    ppl = calculatePerplexity(formatted, model, tokenizer)
    ppls.append(ppl)

df_ppl = pd.DataFrame({'Sequence': sequences_with_ppl, 'Perplexity': ppls})
df_ppl.to_csv(args.output_dir + '/all_generated_with_perplexity.csv', index=False)

k = args.num_return_sequences // 3
top_prots = np.array(sequences_with_ppl)[np.argsort(ppls)[:k]]
top_prots = [i for i in top_prots if len(i) != 0]
print(top_prots)

def remove_prots_with(lst, chars_to_exclude):
    return [s for s in lst if not any(char in chars_to_exclude for char in s)]

print('Ordered by Perplexity.')

data = np.load('./hull_equations.npz')
df_valid = pd.DataFrame({'Sequence': list(top_prots)})

def create_3d_point(sequence, c):
    for ch in ['X','U','B','Z','O','J']:
        sequence = sequence.replace(ch, '')
    count_c = sequence.count(c)
    if count_c == 0: return [0, 0, 0]
    indices = [i for i, ch in enumerate(sequence) if ch == c]
    mean = sum(indices) / count_c
    var = sum((idx - mean) ** 2 for idx in indices) / count_c
    return [count_c, mean, var / len(sequence)]

def point_in_hull(point, hull, tol=1e-12):
    return all(np.dot(eq[:-1], point) + eq[-1] <= tol for eq in hull)

def check_validity(seq):
    return all(point_in_hull(create_3d_point(seq, c), data[c]) for c in data.files)

print('Starting Protein Validity Check...')
df_valid['Valid Protein'] = [check_validity(seq) for seq in top_prots]
filtered_prots = [seq for seq in top_prots if check_validity(seq)]

del model, protgpt2
gc.collect()
torch.cuda.empty_cache()

print('Starting structure check...')
esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
plddt_results = []

for sequence in tqdm(top_prots):
    clean_seq = sequence.replace('\n', '')
    if len(clean_seq) == 0: continue
    inputs = esm_tokenizer([clean_seq], return_tensors="pt", add_special_tokens=False).to(device)
    outputs = esm_model(**inputs)
    plddt_results.append(torch.mean(outputs.plddt).item())
    torch.cuda.empty_cache()

df_valid['plDDT'] = plddt_results
plddt_results = np.array(plddt_results)
df_valid['Good Structure'] = plddt_results > 0.7
df_valid.to_csv(args.output_dir + '/generation_checks.csv', index=False)

plddt_results = np.array([plddt_results[i] for i in range(len(top_prots)) if check_validity(top_prots[i])])
filtered_prots = np.array(filtered_prots)
best_prots = filtered_prots[plddt_results > 0.7]

print('Structure check complete.')

del esm_model
gc.collect()
torch.cuda.empty_cache()

from PeptideBERT.data.dataloader import load_data
from PeptideBERT.model.network import create_model, cri_opt_sch
from PeptideBERT.model.utils import train, validate, test

save_dir = args.pred_model_path
config = yaml.load(open(save_dir + '/config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device
peptideBERT_model = create_model(config)
peptideBERT_model.load_state_dict(torch.load(f'{save_dir}/model.pt')['model_state_dict'], strict=False)

m2 = dict(zip(
    ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','L',
    'A','G','V','E','S','I','K','R','D','T','P','N',
    'Q','F','Y','M','H','C','W','X','U','B','Z','O'], range(30)
))

def f(seq):
    seq = map(lambda x: m2[x], seq)
    seq = torch.tensor([*seq], dtype=torch.long).unsqueeze(0).to(device)
    attn = torch.tensor(seq > 0, dtype=torch.long).to(device)
    return seq, attn

df_preds = pd.DataFrame()
scores = {}
print('Starting protein property predictions...')
best_prots = remove_prots_with(best_prots, "XUBZOJ")

for seq in tqdm(best_prots):
    seq = seq.replace('\n', '')
    scores[seq] = peptideBERT_model(*f(seq)).item()

print('Property prediction completed.')
df_preds['Sequence'] = scores.keys()
df_preds['Score'] = scores.values()
df_preds['Property'] = df_preds['Score'] > 0.5

total_with_property = np.sum(np.array(list(scores.values())) > 0.5)
probability = total_with_property / len(best_prots)
df_preds.to_csv(args.output_dir + '/predictions.csv', index=False)

print('Inference Complete')
with open(args.output_dir + '/info.txt', 'w') as f:
    f.write(f'Total generated: {args.num_return_sequences}, top {args.num_return_sequences//3} by perplexity\n')
    f.write(f'Passed hull: {len(filtered_prots)}, rejected: {len(top_prots) - len(filtered_prots)}\n')
    f.write(f'Passed plddt > 0.7: {len(best_prots)}, {len(best_prots)/len(filtered_prots)*100:.3f}%\n')
    f.write(f'{total_with_property}/{len(best_prots)} had desired property ({probability:.4f})\n')
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")

print(f'Total generated: {args.num_return_sequences}, top {args.num_return_sequences//3} by perplexity')
print(f'Passed hull: {len(filtered_prots)}, rejected: {len(top_prots) - len(filtered_prots)}')
print(f'Passed plddt > 0.7: {len(best_prots)}, {len(best_prots)/len(filtered_prots)*100:.3f}%')
print(f'{total_with_property}/{len(best_prots)} had desired property ({probability:.4f})')
