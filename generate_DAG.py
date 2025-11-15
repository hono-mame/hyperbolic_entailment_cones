import sqlite3
import pandas as pd
import networkx as nx
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="WordNetJpnからハイパーニム・ハイポニム関係を抽出し、DAG形式で保存します。"
    )
    parser.add_argument("n_lines", type=int, help="抽出する行数。全行を使用する場合は0")
    parser.add_argument("filter_nouns", type=lambda x: x.lower() == 'true', help="名詞のみに絞るか (True/False)")
    parser.add_argument("output_dir", type=str, help="出力ディレクトリ")
    parser.add_argument("--root", type=str, default="entity", help="最上位ノード(root)のID")
    parser.add_argument("--output_file", type=str, default="dag_output.tsv", help="出力TSVファイル名")
    return parser.parse_args()

# DAGに変換し、TSVで保存
def csv_to_dag(df, output_tsv, root_node="entity"):
    edges = list(zip(df['hypo'], df['hyper']))  # child -> parent
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    G.add_node(root_node)

    removed_edges = 0
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = next(nx.simple_cycles(G))
        except StopIteration:
            break
        if len(cycle) < 2:
            u = cycle[0]
            if u != root_node:
                G.remove_edge(u, u)
                removed_edges += 1
                print(f"Removed self-loop: {u}->{u}")
        else:
            u, v = cycle[0], cycle[1]
            if root_node not in (u, v):
                G.remove_edge(u, v)
                removed_edges += 1
                print(f"Cycle detected: {cycle}. Removed edge {u}->{v}")
            else:
                print(f"Cycle involves root, skipped: {cycle}")
    
    print(f"Removed {removed_edges} edges to make DAG")

    # 各連結成分を root_node に接続
    for comp in nx.weakly_connected_components(G):
        comp_nodes = set(comp)
        if root_node in comp_nodes:
            continue
        comp_subgraph = G.subgraph(comp_nodes)
        roots = [n for n in comp_subgraph.nodes if comp_subgraph.in_degree(n) == 0]
        for r in roots:
            G.add_edge(r, root_node)
            print(f"Connecting component root {r} -> {root_node}")

    # TSV 出力
    with open(output_tsv, 'w', encoding='utf-8') as f:
        for u, v in G.edges():
            f.write(f"{u}\t{v}\n")

    print(f"DAG ready: {output_tsv}, Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def main():
    args = parse_arguments()
    n_head = args.n_lines
    filter_nouns = args.filter_nouns
    base_path = args.output_dir
    root_node = args.root
    output_file = args.output_file

    if not os.path.isdir(base_path):
        os.makedirs(base_path, exist_ok=True)
        print(f"ディレクトリ作成: {base_path}")

    db_path = "data/wnjpn.db" 
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        print(f"DB接続エラー: {e}")
        return

    query = """
    SELECT 
        w1.lemma AS hyper,
        w2.lemma AS hypo
    FROM synlink AS sl
    INNER JOIN synset AS sy1 ON sy1.synset = sl.synset1
    INNER JOIN synset AS sy2 ON sy2.synset = sl.synset2
    INNER JOIN sense AS se1 ON se1.synset = sy1.synset
    INNER JOIN sense AS se2 ON se2.synset = sy2.synset
    INNER JOIN word AS w1 ON w1.wordid = se1.wordid
    INNER JOIN word AS w2 ON w2.wordid = se2.wordid
    WHERE sl.link = 'hypo'
      AND se1.lang = 'jpn' AND se2.lang = 'jpn'
      AND w1.lang = 'jpn' AND w2.lang = 'jpn'
    """
    if filter_nouns:
        query += " AND sy1.pos = 'n' AND sy2.pos = 'n'"

    df = pd.read_sql_query(query, conn)
    conn.close()

    if n_head > 0 and n_head < len(df):
        df = df.head(n_head)
        print(f"抽出: 先頭 {n_head} 行を使用")
    else:
        print(f"抽出: 全 {len(df)} 行を使用")

    output_tsv = os.path.join(base_path, output_file)
    csv_to_dag(df, output_tsv, root_node=root_node)

if __name__ == "__main__":
    main()
