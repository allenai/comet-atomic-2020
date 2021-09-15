import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from utils import read_jsonl, remove_prefix
from evaluation.eval import QGEvalCap
from tabulate import tabulate


def get_refs_preds(l, type=1):
    if type==1:
        tails = l["fact"]["tails"]
        head = l["fact"]["head"]
        prompt = l["prompt"]
        generations = l["generations"]
        gens = [remove_prefix(g, prompt).strip() for g in generations]
    if type==2:
        tails = l["tails"]
        head = l["head"]
        gens = l["generations"]
    if type==3:
        tails = l["fact"]["tails"]
        head = l["fact"]["head"]
        gens = l["generations"]
    if type==4:
        tails = l["target"]
        head = l["source"]
        gens = l["generations"]
    return gens, tails, head

def get2(l):
    return list(zip(*l))[1]


def topk_eval(model_name, data, data_type, k):
    topk_gts = {}
    topk_res = {}
    topk_exact_match = []
    topk_exact_match_not_none = []
    topk_bleu_score = []

    topk_is_head = []

    for i, l in enumerate(data):
        (gens, tails, head) = get_refs_preds(l, type=data_type)

        sentence_tails = [t.lower() for t in tails]
        split_tails = [t.lower().split() for t in tails]

        for (j, g) in enumerate(gens[:k]):
            key = str(i) + "_" + str(j)
            topk_gts[key] = sentence_tails
            topk_res[key] = [g.lower()]

            b = sentence_bleu(split_tails, g.lower().split(), weights=(0.5, 0.5))
            topk_bleu_score.append((l, b))
            if g in sentence_tails:
                topk_exact_match.append((l, 1))
                if g != "none":
                    topk_exact_match_not_none.append((l, 1))
            else:
                topk_exact_match.append((l, 0))
                if g != "none":
                    topk_exact_match_not_none.append((l, 0))
            if g == head:
                topk_is_head.append((l, 1))
            else:
                topk_is_head.append((l, 0))

    print("---------------TOP K={}---------------".format(k))
    #print(np.mean(get2(topk_exact_match)))
    #print(np.mean(get2(topk_exact_match_not_none)))
    #print(np.mean(get2(topk_bleu_score)))
    QGEval = QGEvalCap(model_name, topk_gts, topk_res)
    scores = QGEval.evaluate()
    scores["Exact_match"] = np.mean(get2(topk_exact_match))
    #scores["TailIsHead"] = np.mean(get2(topk_is_head))
    print(scores)
    return scores


def eval(data_file, data_type, model_name):

    data = read_jsonl(data_file)

    return topk_eval(model_name, data, data_type, k=1)

def toRow(name, results, columns):
    return [name] + [format(float(results[c]), '#.3f') for c in columns]

def main():
    parser = argparse.ArgumentParser()

    #parser.add_argument('--input-file', type=str, help='Dataset filename', default='./data/atomic2020-zeroshot-generations.jsonl')
    parser.add_argument('--input-file', type=str, help='Dataset filename', default='./data/BART/BART-conceptnet.json')
    parser.add_argument('--type', type=int, help="Dataset type", default=1)
    parser.add_argument('--name', type=str, help="Name to refer to dataset", default="")
    args = parser.parse_args()

    # Eval
    expts = [
        [args.input_file, args.name, args.type]
    ]

    add_column = True
    for (f, m, t) in expts:
        s = eval(f, data_type=t, model_name=m)
        columns = list(s.keys())
        s_row = toRow(m, s, columns)
        if add_column:
            rows = [[""] + columns]
            add_column = False
        rows.append(s_row)

    print(tabulate(rows, headers='firstrow', tablefmt='latex', floatfmt='#.3f'))

if __name__ == "__main__":
    main()
