import pandas as pd
import os
import glob

# --- 設定パラメータ ---
# EmbeddingのCSVファイルが存在するディレクトリ
input_dir = "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/data_dim10/models"
# 結果を保存するディレクトリ
output_dir = "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/data_dim10/processed"
# ラベルファイルへのパス (全てのEmbeddingで共通)
label_file_path = "/Users/honokakobayashi/dev/Univ/hyperbolic_entailment_cones/Haskell/data_dim10/input/dag2_all_closure.tsv.vocab"
# Embeddingファイル名のパターン
embedding_file_pattern = os.path.join(input_dir, "*.csv")
# --------------------

# 出力ディレクトリが存在しない場合は作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"出力ディレクトリを作成しました: {output_dir}")

# --- ラベルデータの読み込み (一度だけ実行) ---
try:
    df_label = pd.read_csv(
        label_file_path,
        sep="\t",
        header=None,
        engine="python"
    )
    df_label = df_label.iloc[:, :2]
    df_label.columns = ["word", "label"]
    df_label["word"] = df_label["word"].astype(str)
    print(f"ラベルファイルを読み込みました: {label_file_path}")
except FileNotFoundError:
    print(f"エラー: ラベルファイルが見つかりません: {label_file_path}")
    exit()
except Exception as e:
    print(f"ラベルファイルの読み込み中にエラーが発生しました: {e}")
    exit()

# --- ディレクトリ内の全てのEmbeddingファイルに対して処理を実行 ---
embedding_files = glob.glob(embedding_file_pattern)

if not embedding_files:
    print(f"指定されたパターン '{embedding_file_pattern}' に一致するEmbeddingファイルが見つかりませんでした。")
else:
    print(f"処理対象ファイル数: {len(embedding_files)}")

for file_path in embedding_files:
    file_name = os.path.basename(file_path)
    print(f"\n--- 処理開始: {file_name} ---")

    try:
        # 1. Embeddingデータの読み込み
        df_vec = pd.read_csv(file_path)
        df_vec["word"] = df_vec["word"].astype(str)
        print("Embeddingデータを読み込みました。")

        # 2. マージ
        # 'word'列に基づいて、Embeddingデータとラベルデータを左結合
        df_merged = df_vec.merge(df_label, on="word", how="left")
        print("データフレームをマージしました。")

        # 3. 必要な列の抽出と整形
        # 'dim'で始まる列を抽出
        dim_cols = [c for c in df_merged.columns if c.startswith("dim")]
        
        # 最終データフレームを作成: 'label'列 + 'dim'列
        df_final = df_merged[["label"] + dim_cols]
        # 'label'列の名前を'word'に変更
        df_final = df_final.rename(columns={"label": "word"})
        print("データフレームを整形しました。")

        # 4. 結果の保存
        # 出力ファイル名を作成: 例: 'ep_word_vectors.csv' -> 'processed_ep_word_vectors.csv'
        base_name, ext = os.path.splitext(file_name)
        output_file_name = f"processed_{base_name}{ext}"
        output_file_path = os.path.join(output_dir, output_file_name)

        df_final.to_csv(output_file_path, index=False)
        print(f"結果を保存しました: {output_file_path}")
        print(f"先頭5行:\n{df_final.head()}")

    except Exception as e:
        print(f"ファイル '{file_name}' の処理中にエラーが発生しました: {e}")

print("\n--- 全ての処理が完了しました ---")