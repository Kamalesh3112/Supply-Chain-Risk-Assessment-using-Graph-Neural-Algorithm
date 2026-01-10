[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue.svg)](https://www.python.org/) [![Repo Size](https://img.shields.io/github/repo-size/Kamalesh3112/Supply-Chain-Risk-Assessment-using-Graph-Neural-Algorithm)](https://github.com/Kamalesh3112/Supply-Chain-Risk-Assessment-using-Graph-Neural-Algorithm)

# Supply-Chain Risk Assessment using Graph Neural Algorithm

A Graph Neural Network (GNN) based system for supply chain risk assessment and early warning. This is an **academic and research project** leverages multi-layer graph architectures and an ensemble of GNN models (GCN, GAT, and Temporal GNN) to predict supplier risk, simulate disruption cascades, and provide actionable insights for supply chain resilience.

**Overview**

This repository contains code, configuration examples, and documentation for building and evaluating an ensemble of GNN models to assess risk across supply chains. The system focuses on:

- Supplier risk prediction using node and edge features
- Modeling propagation of disruptions through graph cascades
- Ensemble predictions from GCN, GAT, and Temporal GNN components
- Configurable training and inference pipelines

**Getting started**

1. Install dependencies: pip install -r requirements.txt
2. Prepare data in data/processed (nodes.csv, edges.csv)
3. Train with the example config: python train.py --config configs/train.yaml
4. Run inference with the example config: python inference.py --config configs/inference.yaml

**Files added**

- README.md (this file)
- LICENSE (MIT)
- requirements.txt
- configs/train.yaml
- configs/inference.yaml
- Libraries.py
- Supply Chain Risk Predictor using GNN and LSTM.py
- inference.yaml
- train.yaml

**Contributing**

Contributions, issues and feature requests are welcome. Please open an issue or submit a pull request.

**License**

This project is licensed under the MIT License - see the LICENSE file for details.

(Full README content attached in commit)
