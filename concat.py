import pandas as pd

df_vec = pd.read_csv(
    "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/test_data_10000/task-90percent_dim-5_class-HypCones_init_class-PoincareNIPS_neg_sampl_strategy-true_neg_non_leaves_lr-0.0001_epochs-300_opt-exp_map_where_not_to_sample-ancestors_neg_edges_attach-child_lr_init-0.03_ep_word_vectors.csv"
)

df_label = pd.read_csv(
    "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/test_data_10000/jp_nouns_head_10000_closure.tsv.vocab",
    sep="\t",
    header=None,
    engine="python"
)

df_label = df_label.iloc[:, :2]
df_label.columns = ["word", "label"]

df_vec["word"] = df_vec["word"].astype(str)
df_label["word"] = df_label["word"].astype(str)

df_merged = df_vec.merge(df_label, on="word", how="left")

dim_cols = [c for c in df_merged.columns if c.startswith("dim")]

df_final = df_merged[["label"] + dim_cols]
df_final = df_final.rename(columns={"label": "word"})

df_final.to_csv(
    "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/test_data_10000/embeddings_10000_dim-5.csv",
    index=False
)

print(df_final.head())
