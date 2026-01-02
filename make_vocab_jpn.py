#!/usr/bin/env python3

import nltk
from nltk.corpus import wordnet as wn

# 必要なリソース確認
try:
    wn.synset('entity.n.01')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def synset_id_to_japanese(synset_id: str):
    """
    synset ID に対応する日本語 lemma があればそれを返す。
    なければ None を返す。
    """
    try:
        syn = wn.synset(synset_id)
    except Exception:
        return None

    lemmas_ja = syn.lemma_names(lang='jpn')
    if len(lemmas_ja) > 0:
        return lemmas_ja[0]
    else:
        return None


# ===== 設定 =====
input_path = '/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/F1/noun_closure.tsv.vocab'
output_path = '/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/F1/jn_noun_closure.tsv.vocab'
# =================

with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
    for line in fin:
        line = line.rstrip('\n')
        if not line:
            continue

        idx, synset_id = line.split('\t', 1)
        jp = synset_id_to_japanese(synset_id)

        # 日本語が存在する行のみ書き出す
        if jp is not None:
            fout.write(f'{idx}\t{jp}\n')

print(f'Written: {output_path}')
