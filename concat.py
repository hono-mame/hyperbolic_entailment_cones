import pandas as pd

df_vec = pd.read_csv(
    "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/F1/dim10_90_hypCones.csv",
)

df_label = pd.read_csv(
    "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/F1/jn_noun_closure.tsv.vocab",
    sep="\t",
    header=None,
    engine="python"
)

df_label = df_label.iloc[:, :2]
df_label.columns = ["word", "label"]

df_vec["word"] = df_vec["word"].astype(str)
df_label["word"] = df_label["word"].astype(str)

df_merged = df_vec.merge(df_label, on="word", how="inner")

dim_cols = [c for c in df_merged.columns if c.startswith("dim")]

df_final = df_merged[["label"] + dim_cols].rename(
    columns={"label": "word"}
)

df_final.to_csv(
    "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/F1/Embedding.csv",
    index=False
)

print(df_final.head())
