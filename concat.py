import pandas as pd
import nltk
from nltk.corpus import wordnet as wn

try:
    wn.synset('entity.n.01')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


vec_path = "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/F1/dim10_90_hypCones.csv"
vocab_en_path = "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/F1/noun_closure.tsv.vocab"
output_path = "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/F1/Embedding.csv"

df_vocab_en = pd.read_csv(vocab_en_path, sep="\t", header=None, names=["id", "synset_id"])

ja_data = []
for _, row in df_vocab_en.iterrows():
    idx = str(row["id"])
    sid = row["synset_id"]
    try:
        syn = wn.synset(sid)
        lemmas_ja = syn.lemma_names(lang='jpn')
        if lemmas_ja:
            for jp in set(lemmas_ja):  # 重複を除いて追加
                ja_data.append({"word": idx, "label_ja": jp})
    except:
        continue

df_label_ja = pd.DataFrame(ja_data)


df_vec = pd.read_csv(vec_path)
df_vec["word"] = df_vec["word"].astype(str)

df_merged = df_vec.merge(df_label_ja, on="word", how="inner")

dim_cols = [c for c in df_merged.columns if c.startswith("dim")]
df_final = df_merged[["label_ja"] + dim_cols].rename(columns={"label_ja": "word"})


df_final.to_csv(output_path, index=False)

print(f"変換完了。保存先: {output_path}")
print(df_final.head())