/**
 * Brandes' algorithm for betweenness centrality in unweighted graphs.
 * Based on: Ulrik Brandes, "A Faster Algorithm for Betweenness Centrality",
 * Journal of Mathematical Sociology 25(2):163-177, 2001.
 *
 * For an undirected graph the final scores must be divided by 2 because
 * each shortest path is counted twice (once in each direction).
 */

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <limits>
#include <algorithm>

using namespace std;

/**
 * Compute betweenness centrality for all vertices in an unweighted graph.
 *
 * @param adj       adjacency list: adj[v] = list of neighbors of vertex v
 * @param directed  true if graph is directed, false for undirected
 * @return          vector of betweenness centrality scores (one per vertex)
 */
vector<double> brandesBetweenness(const vector<vector<int>>& adj, bool directed) {
    int n = adj.size();                     // number of vertices
    vector<double> CB(n, 0.0);              // final betweenness scores

    // For each vertex as source
    for (int s = 0; s < n; ++s) {
        // ---- Phase 1: BFS from s to compute distances and shortest path counts ----
        stack<int> S;                       // vertices in order of non‑increasing distance
        vector<vector<int>> P(n);           // list of predecessors on shortest paths
        vector<int> sigma(n, 0);            // number of shortest paths from s to v
        vector<int> dist(n, -1);            // distance from s to v

        sigma[s] = 1;
        dist[s] = 0;

        queue<int> Q;
        Q.push(s);

        while (!Q.empty()) {
            int v = Q.front(); Q.pop();
            S.push(v);

            for (int w : adj[v]) {
                // First time we see w?
                if (dist[w] < 0) {
                    dist[w] = dist[v] + 1;
                    Q.push(w);
                }
                // Is v on a shortest path from s to w?
                if (dist[w] == dist[v] + 1) {
                    sigma[w] += sigma[v];
                    P[w].push_back(v);
                }
            }
        }

        // ---- Phase 2: Accumulate dependencies (reverse order) ----
        vector<double> delta(n, 0.0);       // dependency of s on v

        while (!S.empty()) {
            int w = S.top(); S.pop();
            for (int v : P[w]) {
                // Dependency accumulation formula (paper, page 10)
                delta[v] += (static_cast<double>(sigma[v]) / sigma[w]) * (1.0 + delta[w]);
            }
            if (w != s) {
                CB[w] += delta[w];
            }
        }
    }

    // For undirected graphs, each path is counted twice; correct by dividing by 2
    if (!directed) {
        for (int i = 0; i < n; ++i) {
            CB[i] /= 2.0;
        }
    }

    return CB;
}

// ---------------------------------------------------------------------
// Example usage
// ---------------------------------------------------------------------

int main() {
    // Build a simple undirected graph (the "triangle with a tail" example)
    // Vertices: 0 -- 1 -- 2 -- 3
    //            \        /
    //              4
    int n = 5;
    vector<vector<int>> adj(n);

    // Undirected edges
    adj[0].push_back(1);
    adj[0].push_back(4);
    adj[1].push_back(0);
    adj[1].push_back(2);
    adj[2].push_back(1);
    adj[2].push_back(3);
    adj[2].push_back(4);
    adj[3].push_back(2);
    adj[4].push_back(0);
    adj[4].push_back(2);

    // Compute betweenness centrality (undirected)
    vector<double> bc = brandesBetweenness(adj, false);

    // Print results
    cout << "Betweenness Centrality (undirected graph):" << endl;
    for (int i = 0; i < n; ++i) {
        cout << "Vertex " << i << ": " << bc[i] << endl;
    }

    // If the graph were weighted, Dijkstra would replace BFS.
    // The core dependency accumulation step (Phase 2) remains identical.

    return 0;
}
