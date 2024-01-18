from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score as meteor_scorer
from nltk.tokenize import wordpunct_tokenize
import json
from bert_score import score
from tqdm.auto import tqdm
import argparse
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')
parser = argparse.ArgumentParser()
parser.add_argument("--ref", default='gt_caption.json')
parser.add_argument("--pred", default='gen_caption.json')
args = parser.parse_args()

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

ref_caption = json.load(open(args.ref, 'r'))
pred_caption = json.load(open(args.pred, 'r'))

pred_list = []
ref_list = []
for k, v in ref_caption.items():
    if k not in pred_caption.keys():
        ValueError()
    ref_list.append(v)
    pred_list.append(pred_caption[k])


def evaluate(model_name, candidates, mult_reference):
    rouge_score, bleu_score, bleu4_score, meteor_score = 0, 0, 0, 0
    for ref, cand in tqdm(zip(mult_reference, candidates), total=len(mult_reference)):
        rouge_score += scorer.score(ref, cand)['rougeL'].recall
        cand_split = wordpunct_tokenize(cand)
        ref_split = wordpunct_tokenize(ref)
        bleu4_score += sentence_bleu([ref], cand, weights=(0.0, 0.0, 0.0, 1.0))
        bleu_score += sentence_bleu([ref], cand)
        meteor_score += meteor_scorer([ref_split], cand_split)
        print(rouge_score, bleu_score, bleu4_score, meteor_score)
    rouge_score, bleu_score, bleu4_score, meteor_score = rouge_score / (len(candidates)), bleu_score / (len(candidates)), bleu4_score / (len(candidates)), meteor_score / (len(candidates))
    P, R, F1 = score(candidates, mult_reference, lang="en", verbose=True)
    bert_score = R.mean().item()
    print(f"Model: {model_name}")
    print(f"BLEU Score: {bleu_score}")
    print(f"BLEU-4 Score: {bleu4_score}")
    print(f"METEOR Score: {meteor_score}")
    print(f"ROUGE Score: {rouge_score}")
    print(f"BERT Score: {bert_score}")


evaluate("UniMuMo", candidates=pred_list, mult_reference=ref_list)

