#include <iostream>
#include <stack>
#include <vector>
#include <queue>
#include <algorithm>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

namespace py = pybind11;

// Structure to hold an edge with a weight (distance/travel time)
struct Edge {
    int target;
    double weight;
};

class GraphBetweennessCentrality {
    int numberOfVertices;
    std::vector<std::vector<Edge>> adjacencyList;
    bool isDirected;

public:
    explicit GraphBetweennessCentrality(int _n, bool isDirected) {
        adjacencyList.resize(_n);
        numberOfVertices = _n;
        this->isDirected = isDirected;
    }

    // Now accepts a weight parameter for road distances
    void addEdge(int u, int v, double weight) {
        adjacencyList[u].push_back({v, weight});
        if (!isDirected) {
            adjacencyList[v].push_back({u, weight});
        }
    }

    std::vector<double> betweennessCentrality() {
        std::vector<double> CB(numberOfVertices, 0.0);

        /* OpenMP pragma to parallelize the loop across all CPU cores! */
        #pragma omp parallel for
        for (int source = 0; source < numberOfVertices; source += 1) {
            std::stack<int> Stack;
            std::vector<std::vector<int>> P(numberOfVertices);
            std::vector<double> sigma(numberOfVertices, 0.0);
            
            // distance array changed to double to handle precise lengths
            std::vector<double> distance(numberOfVertices, std::numeric_limits<double>::infinity());

            sigma[source] = 1.0;
            distance[source] = 0.0;

            // Upgrade: Replaced standard queue with a Priority Queue (Min-Heap) for Dijkstra's
            using NodeDist = std::pair<double, int>;
            std::priority_queue<NodeDist, std::vector<NodeDist>, std::greater<NodeDist>> pq;
            
            pq.push({0.0, source});

            while (!pq.empty()) {
                double dist = pq.top().first;
                int vertex = pq.top().second;
                pq.pop();

                // Skip outdated distance calculations
                if (dist > distance[vertex]) continue;

                Stack.push(vertex);

                for (const Edge& edge : adjacencyList[vertex]) {
                    int w = edge.target;
                    double weight = edge.weight;

                    // Relax the edge
                    if (distance[vertex] + weight < distance[w]) {
                        distance[w] = distance[vertex] + weight;
                        pq.push({distance[w], w});
                        sigma[w] = 0.0;
                        P[w].clear();
                    }
                    /* If shortest path is found, accumulate sigma */
                    if (distance[vertex] + weight == distance[w]) {
                        sigma[w] += sigma[vertex];
                        P[w].push_back(vertex);
                    }
                }
            }

            std::vector<double> delta(numberOfVertices, 0.0);
            while (!Stack.empty()) {
                int w = Stack.top(); 
                Stack.pop();
                for (int v : P[w]) {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if (w != source) {
                    /* OpenMP Atomic update to prevent race conditions when writing to the shared CB array */
                    #pragma omp atomic
                    CB[w] += delta[w];
                }
            }
        }

        if (!isDirected) {
            for (int i = 0; i < numberOfVertices; ++i) {
                CB[i] /= 2.0;
            }
        }

        return CB;
    }
};

/* PYBIND11 Wrapper */
PYBIND11_MODULE(graph_betweenness_centrality, m) {
    m.doc() = "High-performance C++ Graph Metrics for Python";

    py::class_<GraphBetweennessCentrality>(m, "GraphBetweennessCentrality")
        .def(py::init<int, bool>(), py::arg("n"), py::arg("isDirected"))
        .def("add_edge", &GraphBetweennessCentrality::addEdge, py::arg("u"), py::arg("v"), py::arg("weight"))
        .def("betweenness_centrality", &GraphBetweennessCentrality::betweennessCentrality);
}
