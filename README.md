Route Resilience: Occlusion-Robust Road Extraction & Criticality Analysis

ISRO Hackathon 2026 | Transforming raw satellite imagery into mathematically routable urban networks for disaster response and urban planning.

## 🛑 The Problem

During natural disasters, emergency responders rely on continuous, routable road networks. However, standard satellite mapping fails when roads are occluded by tree canopies, shadows, or heavy cloud cover. Furthermore, traditional AI segmentation outputs static pixels, which are mathematically useless for routing algorithms (like Dijkstra's) or vulnerability analysis.

## 💡 My Solution

Route Resilience is a full-stack geospatial intelligence platform that shifts satellite mapping from passive image processing to active mathematical simulation.

Vision (Spectral Blindness Mitigation): A PyTorch U-Net (ResNet-34 backbone) processes multi-spectral GeoTIFFs, utilizing contextual awareness and a custom Hybrid Dice+BCE Loss function to "hallucinate" roads beneath occlusions.

Topology (Pixels to Graphs): The probability mask is morphologically skeletonized and converted into a topological network of nodes (intersections) and edges (road segments).

Simulation (High-Performance Compute): A custom C++ backend executes Brandes' algorithm to instantly calculate Betweenness Centrality, identifying critical "gatekeeper" bottlenecks.

Action (Interactive Dashboard): Urban planners can simulate real-time disasters (e.g., flash floods, bridge collapses) via a Streamlit/Folium dashboard and instantly visualize the cascading vulnerabilities across the city.

## 🛠️ Technology Stack
```
High-Performance Backend: C++, PyBind11, OpenMP

Deep Learning: PyTorch, Torchvision

Geospatial Processing: Rasterio, Folium, Skimage

Graph Theory: NetworkX

Frontend: Streamlit, Streamlit-Folium
```


## 📂 Project Structure
```text
├── dashboard.py                  # Main Streamlit application & pipeline controller
├── road_segmentation_model.py    # PyTorch U-Net & ResNet-34 architecture
├── BetweennessCentrality.cpp     # High-performance C++ Graph Engine
├── bridge.py                     # AI-to-Graph conversion logic (Testing/Standalone)
├── model.pth                     # Trained PyTorch weights (Download required)
└── README.md                     # Project documentation
```
