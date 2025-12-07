import os
import numpy as np
import sqlite3
import random
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, Set, List, Tuple

def get_english_labels(db_path: str) -> Dict[str, str]:
    """DBから英語のLemmaをSynset IDに紐づけて取得する"""
    if not os.path.exists(db_path):
        return {}
    
    print("Loading English labels...")
    conn = sqlite3.connect(db_path)
    query = """
    SELECT s.synset, w.lemma
    FROM sense s
    JOIN word w ON s.wordid = w.wordid
    WHERE w.lang = 'eng'
    GROUP BY s.synset
    ORDER BY s.synset, s.rank ASC
    """
    cursor = conn.cursor()
    cursor.execute(query)
    # Synset ID -> English Label
    synset2english = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return synset2english


def get_japanese_labels(db_path: str) -> Dict[str, str]:
    if not os.path.exists(db_path):
        print(f"Error: DB file not found at {db_path}.")
        return {}

    print(f"Loading Japanese labels from {db_path}...")
    conn = sqlite3.connect(db_path)

    query = """
    SELECT s.synset, w.lemma
    FROM sense s
    JOIN word w ON s.wordid = w.wordid
    WHERE w.lang = 'jpn'
    GROUP BY s.synset
    """

    cursor = conn.cursor()
    cursor.execute(query)
    synset2label = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()

    print(f"Loaded {len(synset2label)} Japanese labels.")
    return synset2label


def create_all_data(full_edges_data_file: str, root_str: str, db_path: str):

    # --- ラベル辞書のロード ---
    synset2jpn = get_japanese_labels(db_path)
    synset2eng = get_english_labels(db_path) # 英語ラベルを追加

    # --- ラベル取得ヘルパー関数 ---
    def get_preferred_label(synset_id: str) -> str:
        """日本語 -> 英語 -> Synset ID の順にラベルを返す"""
        if synset_id in synset2jpn:
            return synset2jpn[synset_id]
        if synset_id in synset2eng:
            return synset2eng[synset_id]
        return synset_id


    # === 1. 全ノードの読み込みとID割り振り ===
    print("Loading all nodes for initial mapping...")
    all_nodes_list = []
    with open(full_edges_data_file, 'r') as f:
        for line in f:
            all_nodes_list.extend(line.split())
    
    all_nodes = list(set(all_nodes_list))
    # 全ノードが最終的なノードセットとなる
    idx2node_all = sorted(all_nodes) # 安定させるためにソート
    node2idx_all = {node: idx for idx, node in enumerate(idx2node_all)}
    
    num_all_nodes = len(all_nodes)
    print("Num final nodes =", num_all_nodes)


    # === 2. エッジ読み込みと推移閉包 (全ノードで計算) ===
    outgoing_edges_all = defaultdict(set)
    ingoing_edges_all = defaultdict(set)
    all_edges_synset: List[Tuple[str, str]] = [] # フィルタリング前の Synset ペアリスト

    print("Reading and mapping all edges...")
    with open(full_edges_data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) < 2: continue

            child, parent = parts[0], parts[1]
            if child == parent or child == root_str: continue 

            c = node2idx_all.get(child)
            p = node2idx_all.get(parent)
            if c is None or p is None: continue 

            outgoing_edges_all[p].add(c)
            ingoing_edges_all[c].add(p)
    
    # --- 推移閉包の計算 ---
    print(f"Computing Transitive Closure on ALL {num_all_nodes} nodes...")
    for k in tqdm(range(num_all_nodes)):
        i_nodes = list(ingoing_edges_all[k])
        j_nodes = outgoing_edges_all[k]
        if not j_nodes: continue
        for i in i_nodes:
            new_edges = j_nodes - outgoing_edges_all[i]
            if new_edges:
                outgoing_edges_all[i].update(new_edges)
                for j in new_edges:
                    ingoing_edges_all[j].add(i)

    # === 3. フィルタリングなしで全エッジを確定し、新しいIDに再マッピング ===
    
    # フィルタリングされたエッジリストを格納 (ID形式)
    all_edges_indices: List[Tuple[int, int]] = []
    
    for p_idx in range(num_all_nodes):
        for c_idx in outgoing_edges_all[p_idx]:
            all_edges_indices.append((p_idx, c_idx))
            
            # Synsetペアも収集 (後のラベル出力用)
            parent_synset = idx2node_all[p_idx]
            child_synset = idx2node_all[c_idx]
            all_edges_synset.append((parent_synset, child_synset))

    # --- ここから、元のコードのロジックを新しいIDとファイルパスで再開 ---
    
    # 新しい出力ファイル名の設定
    new_base_file_path = full_edges_data_file.replace(".tsv", "") + "_closure.tsv"
    
    # ===== 5. フィルタリングなし推移閉包のファイル保存 (ラベル付き) =====
    print(f"Writing full closure (Multi-lingual labels) to {new_base_file_path}...")
    with open(new_base_file_path, 'w') as f:
        for parent_synset, child_synset in all_edges_synset:
            parent_label = get_preferred_label(parent_synset)
            child_label = get_preferred_label(child_synset)
            # WordNet形式 (Child \t Parent)
            f.write(f"{child_label}\t{parent_label}\n") 


    # ===== 6. vocab（ID + ラベル・2列）=====
    print("Writing vocab (ID + Preferred label)...")
    with open(new_base_file_path + '.vocab', 'w') as f:
        # idx2node_all はソート済み全ノードリスト
        for idx, node_synset in enumerate(idx2node_all):
            label = get_preferred_label(node_synset)
            f.write(f"{idx}\t{label}\n")


    # ===== 7. 推移的簡約 (全ノード/新しいIDで計算) =====
    print("Computing Transitive Reduction (All nodes)...")

    # 処理継続のため、outgoing_edges_jpn -> outgoing_edges_all に名称を変更
    outgoing_edges = outgoing_edges_all
    ingoing_edges = ingoing_edges_all
    num_nodes = num_all_nodes

    basic_outgoing_edges = {i: outgoing_edges[i].copy() for i in range(num_nodes)}
    basic_ingoing_edges = {i: ingoing_edges[i].copy() for i in range(num_nodes)}

    for k in tqdm(range(num_nodes)):
        for i in ingoing_edges[k]:
            for j in outgoing_edges[k]:
                basic_outgoing_edges[i].discard(j)
                basic_ingoing_edges[j].discard(i)

    all_edges_non_basic_indices = []
    all_edges_basic_indices = []

    for i in range(num_nodes):
        for j in outgoing_edges[i]:
            if j not in basic_outgoing_edges[i]:
                all_edges_non_basic_indices.append((i, j))
            else:
                all_edges_basic_indices.append((i, j))

    print(f"Num Basic Edges (TR): {len(all_edges_basic_indices)}")
    print(f"Num Non-Basic Edges (TC): {len(all_edges_non_basic_indices)}")

    # ===== 8. データ分割と負例生成の準備 =====
    print("Preparing optimized negative sampling...")

    # filtered_outgoing/ingoing は全エッジ (TC) を含んでいる
    filtered_outgoing = outgoing_edges
    filtered_ingoing = ingoing_edges
    valid_node_indices = np.arange(num_nodes)

    def gen_negs_fast(node_idx, excluded, num_neg=5):
        excluded = set(excluded)
        excluded.add(node_idx)

        candidates = np.setdiff1d(
            valid_node_indices,
            np.fromiter(excluded, dtype=np.int32),
            assume_unique=False
        )
        if len(candidates) == 0:
            return []
        if len(candidates) >= num_neg:
            return np.random.choice(candidates, size=num_neg, replace=False).tolist()
        else:
            return np.random.choice(candidates, size=num_neg, replace=True).tolist()

    def gen_and_write_negs_fast(pos_edges_indices, file, desc="Gen Negs"):
        buffer = []
        # num_nodes_jpn -> num_nodes に変更
        for (parent_idx, child_idx) in tqdm(pos_edges_indices, desc=desc):
            if len(filtered_outgoing[parent_idx]) < num_nodes - 1:
                negatives = gen_negs_fast(parent_idx, filtered_outgoing[parent_idx])
                for neg_idx in negatives:
                    buffer.append(f"{parent_idx}\t{neg_idx}\n")

            negatives = gen_negs_fast(child_idx, filtered_ingoing[child_idx])
            for neg_idx in negatives:
                buffer.append(f"{neg_idx}\t{child_idx}\n")
        file.writelines(buffer)
    
    # ===== 9. 補助ファイルと分割ファイルの出力 (すべて新しいファイル名) =====
    
    # .full_transitive (IDのみ)
    with open(new_base_file_path + '.full_transitive', 'w') as f:
        for p, c in all_edges_indices:
            f.write(f"{p}\t{c}\n")

    # .full_neg
    with open(new_base_file_path + '.full_neg', 'w') as f:
        gen_and_write_negs_fast(all_edges_non_basic_indices, f, "Full Negatives")

    # データ分割
    train_perc_list = [0, 10, 25, 50, 90]
    valid_perc = 5
    test_perc = 5

    random.shuffle(all_edges_non_basic_indices)

    test_split_idx = int(test_perc * len(all_edges_non_basic_indices) / 100.0)
    valid_split_idx = int(valid_perc * len(all_edges_non_basic_indices) / 100.0)

    test_pairs_idx = all_edges_non_basic_indices[:test_split_idx]
    valid_pairs_idx = all_edges_non_basic_indices[test_split_idx:test_split_idx + valid_split_idx]
    remaining_pairs_idx = all_edges_non_basic_indices[test_split_idx + valid_split_idx:]

    def write_pairs_id_only(fname, pairs):
        with open(fname, 'w') as f:
            for p, c in pairs:
                f.write(f"{p}\t{c}\n")

    write_pairs_id_only(new_base_file_path + '.test', test_pairs_idx)
    write_pairs_id_only(new_base_file_path + '.valid', valid_pairs_idx)

    with open(new_base_file_path + '.test_neg', 'w') as f:
        gen_and_write_negs_fast(test_pairs_idx, f, "Test Negatives")

    with open(new_base_file_path + '.valid_neg', 'w') as f:
        gen_and_write_negs_fast(valid_pairs_idx, f, "Valid Negatives")

    for train_perc in train_perc_list:
        train_count = int(train_perc * len(all_edges_non_basic_indices) / 100.0)
        current_train_sample = random.sample(
            remaining_pairs_idx,
            min(train_count, len(remaining_pairs_idx))
        )

        filename = new_base_file_path + f'.train_{train_perc}percent'
        with open(filename, 'w') as f:
            for p_idx, c_idx in current_train_sample:
                f.write(f"{p_idx}\t{c_idx}\n")
            for p_idx, c_idx in all_edges_basic_indices:
                f.write(f"{p_idx}\t{c_idx}\n")


current_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(current_directory, 'data', 'maxn')

input_dag_file = os.path.join(data_directory, 'dag2_all.tsv')
db_file_path = os.path.join(data_directory, 'wnjpn.db')
root_synset_id = '00001740-n'

if os.path.exists(input_dag_file) and os.path.exists(db_file_path):
    create_all_data(input_dag_file, root_synset_id, db_file_path)
else:
    print("Error: Files not found.")