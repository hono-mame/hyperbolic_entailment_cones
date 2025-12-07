#include <sqlite3.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <queue>
#include <algorithm>

using namespace std;
namespace fs = std::filesystem;

struct Edge { int u, v; };

// DFS-based detection of back edges; collect edges to remove
bool dfs_remove_backedges(int node, const vector<vector<int>>& adj, vector<int>& color, unordered_set<long long>& remove_set) {
    color[node] = 1; // gray
    for (int nei : adj[node]) {
        if (color[nei] == 0) {
            if (dfs_remove_backedges(nei, adj, color, remove_set)) return true; 
        } else if (color[nei] == 1) {
            // back edge detected: node -> nei
            long long key = ((long long)node << 32) | (unsigned int)nei;
            remove_set.insert(key);
        }
    }
    color[node] = 2; // black
    return false;
}

int main(int argc, char** argv) {
    // 引数: n_lines, filter_nouns, output_dir, [--root ROOT], [--output_file FILE]
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " n_lines filter_nouns(True|False) output_dir [--root ROOT_ID] [--output_file FILE]" << endl;
        return 1;
    }

    long long n_lines = atoll(argv[1]);
    string filter_str = argv[2];
    bool filter_nouns = false;
    for (auto &c : filter_str) c = tolower(c);
    filter_nouns = (filter_str == "true" || filter_str == "1");
    string output_dir = argv[3];
    
    // WordNet 3.0 / WN-Ja における entity.n.01 のID
    string root_node = "00001740-n"; 
    string output_file = "dag_output.tsv";

    // parse optional args
    for (int i = 4; i + 1 < argc; ++i) {
        string opt = argv[i];
        if (opt == "--root") { root_node = argv[i+1]; ++i; }
        else if (opt == "--output_file") { output_file = argv[i+1]; ++i; }
        else { /* ignore unknown */ }
    }

    fs::create_directories(output_dir);
    string output_tsv = (fs::path(output_dir) / output_file).string();

    // open sqlite
    sqlite3* db = nullptr;
    int rc = sqlite3_open("data/wnjpn.db", &db);
    if (rc != SQLITE_OK) {
        cerr << "Cannot open database: " << sqlite3_errmsg(db) << endl;
        return 1;
    }

    // speed pragmas
    vector<string> pragmas = {
        "PRAGMA journal_mode = MEMORY;",
        "PRAGMA temp_store = MEMORY;",
        "PRAGMA synchronous = OFF;",
        "PRAGMA cache_size = -2000000;"
    };
    for (auto &p : pragmas) sqlite3_exec(db, p.c_str(), nullptr, nullptr, nullptr);

    // インデックス作成 (Synset IDでの検索になるため、synlinkのインデックスが重要)
    sqlite3_exec(db, "CREATE INDEX IF NOT EXISTS idx_sl_s1_s2_link ON synlink(synset1, synset2, link);", nullptr, nullptr, nullptr);
    if (filter_nouns) {
        sqlite3_exec(db, "CREATE INDEX IF NOT EXISTS idx_sy_synset_pos ON synset(synset, pos);", nullptr, nullptr, nullptr);
    }

    // wordテーブル、senseテーブルへの結合を排除。synlinkから直接IDを取得。
    // 日本語かどうかの判定も不要（構造は言語非依存）。
    string query =
        "SELECT sl.synset1 AS hyper, sl.synset2 AS hypo "
        "FROM synlink AS sl ";
    
    // 名詞フィルタが必要な場合のみ synset テーブルを結合
    if (filter_nouns) {
        query += "INNER JOIN synset AS sy1 ON sy1.synset = sl.synset1 ";
        query += "INNER JOIN synset AS sy2 ON sy2.synset = sl.synset2 ";
    }

    query += "WHERE sl.link = 'hypo' ";
    
    if (filter_nouns) {
        query += "AND sy1.pos = 'n' AND sy2.pos = 'n' ";
    }

    cout << "Executing Query: " << query << endl;

    sqlite3_stmt* stmt = nullptr;
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        cerr << "Failed to prepare query: " << sqlite3_errmsg(db) << endl;
        sqlite3_close(db);
        return 1;
    }

    unordered_map<string,int> idmap;
    idmap.reserve(200000); // WordNet Synset数は約10万強
    vector<string> rev;
    rev.reserve(200000);
    vector<Edge> edges;
    edges.reserve(200000);

    auto get_id = [&](const string& s)->int{
        auto it = idmap.find(s);
        if (it != idmap.end()) return it->second;
        int id = (int)rev.size();
        idmap.emplace(s, id);
        rev.push_back(s);
        return id;
    };

    long long rowcount = 0;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const unsigned char* hyper_text = sqlite3_column_text(stmt, 0);
        const unsigned char* hypo_text = sqlite3_column_text(stmt, 1);
        string hyper = hyper_text ? reinterpret_cast<const char*>(hyper_text) : string();
        string hypo  = hypo_text  ? reinterpret_cast<const char*>(hypo_text ) : string();

        // remove self-loop (データ不整合対策)
        if (!hypo.empty() && hypo == hyper) continue;

        // u = Child, v = Parent
        int u = get_id(hypo);
        int v = get_id(hyper);
        edges.push_back({u,v});

        ++rowcount;
        if (n_lines > 0 && rowcount >= n_lines) break;
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    cout << "抽出完了。総行数: " << edges.size() << "\n";

    // ensure root exists
    int root_id;
    auto itroot = idmap.find(root_node);
    if (itroot == idmap.end()) {
        root_id = get_id(root_node);
        cout << "Root node '" << root_node << "' を追加しました (id=" << root_id << ").\n";
    } else root_id = itroot->second;

    int N = (int)rev.size();
    vector<vector<int>> adj;
    vector<vector<int>> radj;
    adj.assign(N, {});
    radj.assign(N, {});
    vector<int> indeg(N,0);

    // グラフ構築 (u:Child -> v:Parent)
    for (auto &e : edges) {
        if (e.u < 0 || e.v < 0) continue;
        adj[e.u].push_back(e.v);
        radj[e.v].push_back(e.u); // 逆辺
        indeg[e.v]++;
    }

    // サイクル除去
    cout << "--- グラフのDAG化（サイクル除去） ---\n";
    int removed_edges = 0;
    while (true) {
        vector<int> color(N, 0);
        unordered_set<long long> remove_set;
        // Synsetベースならサイクルは少ないはずなので小さめに
        remove_set.reserve(1024); 
        for (int i = 0; i < N; ++i) {
            if (color[i] == 0) dfs_remove_backedges(i, adj, color, remove_set);
        }
        if (remove_set.empty()) break;
        
        // apply removals
        for (auto key : remove_set) {
            int u = (int)(key >> 32);
            int v = (int)(key & 0xFFFFFFFF);
            auto &vec = adj[u];
            auto it = remove(vec.begin(), vec.end(), v);
            if (it != vec.end()) {
                vec.erase(it, vec.end());
                removed_edges++;
            }
            // remove reverse
            auto &rvec = radj[v];
            auto it2 = remove(rvec.begin(), rvec.end(), u);
            if (it2 != rvec.end()) rvec.erase(it2, rvec.end());
            indeg[v] = (int)radj[v].size(); // indeg更新（念のため）
        }
    }

    cout << "サイクル除去により削除されたエッジ数: " << removed_edges << "\n";
    cout << "---------------------------------\n";

    // 連結成分の統合
    cout << "--- 連結成分の統合 ---\n";
    vector<int> comp(N, -1);
    int comp_id = 0;
    for (int i = 0; i < N; ++i) {
        if (comp[i] != -1) continue;
        queue<int> q;
        q.push(i);
        comp[i] = comp_id;
        while (!q.empty()) {
            int x = q.front(); q.pop();
            for (int y : adj[x]) if (comp[y] == -1) { comp[y] = comp_id; q.push(y); }
            for (int y : radj[x]) if (comp[y] == -1) { comp[y] = comp_id; q.push(y); }
        }
        comp_id++;
    }

    int root_comp = comp[root_id];
    vector<vector<int>> comp_nodes(comp_id);
    for (int i = 0; i < N; ++i) comp_nodes[comp[i]].push_back(i);

    int connections_made = 0;
    for (int cid = 0; cid < comp_id; ++cid) {
        if (cid == root_comp) continue;
        
        // ローカルルート（その成分の中で親がいないノード）を探す
        // adj: child -> parent なので、adj[node] が空ではなく、
        // 「グラフ全体での out-degree ではなく、コンポーネント内での out-degree」を見る必要がある
        // しかし、WordNetの構造上、一番上のノードは親を持たない(adjが空)。
        
        vector<int> local_roots;
        for (int node : comp_nodes[cid]) {
            // 親(adj)がコンポーネント内に存在するか確認
            // adj[node] の中に、同じcompIDを持つものがなければ、それはローカルルート
            bool has_parent_in_comp = false;
            for (int p : adj[node]) {
                 if (comp[p] == cid) { has_parent_in_comp = true; break; }
            }
            if (!has_parent_in_comp) local_roots.push_back(node);
        }

        for (int r : local_roots) {
            adj[r].push_back(root_id);
            radj[root_id].push_back(r); // 整合性のため
            connections_made++;
        }
    }
    
    if (connections_made > 0) {
        cout << "孤立コンポーネントから " << connections_made << " 個のローカルルートを " << root_node << " に接続しました。\n";
    }

    cout << "--- グラフのTSV出力 ---\n";
    ofstream ofs(output_tsv, ios::out | ios::binary);
    if (!ofs) {
        cerr << "出力ファイルを開けません: " << output_tsv << endl;
        return 1;
    }
    
    // 出力形式: Child \t Parent
    // adj[u] は u(Child) -> v(Parent)
    long long out_edges = 0;
    for (int u = 0; u < N; ++u) {
        for (int v : adj[u]) {
            ofs << rev[u] << '\t' << rev[v] << '\n';
            out_edges++;
        }
    }
    ofs.close();

    cout << "DAG生成完了: " << output_tsv << "\n";
    cout << "ノード数: " << N << ", エッジ数: " << out_edges << "\n";
    cout << "----------------------\n";
    return 0;
}