# IDS 568 Final Project: Monitoring, Governance & Drift Detection

## Overview
This project implements a production-style MLOps framework for a Retrieval-Augmented Generation (RAG) system. The goal is to demonstrate capabilities in monitoring, experimentation, governance, drift detection, and AI risk assessment.

The system simulates a real-world deployment environment where data quality, system health, and risk management are continuously evaluated.

---

## Project Structure
```
ids568-final-project/
├── src/
│   ├── monitoring/
│   ├── ab_test/
│   └── drift/
│
├── docs/
│   ├── dashboard-interpretation.md
│   ├── experiment-specification.md
│   ├── recommendation-memo.md
│   ├── model-card.md
│   ├── risk-register.md
│   ├── drift-diagnostic-report.md
│   ├── governance-review.md
│   ├── risk-matrix.md
│   ├── cto-memo.md
│
├── visualizations/
├── logs/
├── dashboards/
├── requirements.txt
└── README.md
```

---

## System Architecture
The system follows a RAG pipeline:

User → Retriever → LLM → Tool Use (optional) → Output

- Retriever: Fetches relevant documents using embeddings and vector search  
- LLM: Generates responses using retrieved context  
- Tool Layer: Optional execution layer for additional processing  
- Output: Final response returned to the user  

---

## Setup Instructions (Mac + VS Code)

### 1. Clone Repository
```
git clone <your-repo-url>
cd ids568-final-project
```

### 2. Create Virtual Environment
```
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

---

## How to Run

### Drift Detection
```
python -m src.drift.drift_detection
```

This script:
- Analyzes audit logs
- Computes drift metrics (PSI, KS)
- Generates visualizations
- Outputs drift summary

---

## Key Results

- Retrieval Score PSI: 0.150 (Moderate drift)  
- Query Length PSI: 0.085 (Minimal drift)  
- KS Test p-value: 0.0949 (Not statistically significant)  
- No major performance degradation observed  

Conclusion: No immediate retraining required. Continue monitoring.

---

## Component Summary

### Component 1: Monitoring Dashboard
Tracks latency, throughput, error rate, and drift signals.

### Component 2: A/B Testing
Simulates model comparison with statistical evaluation.

### Component 3: Governance & Model Card
Includes model documentation, lineage, and risk register.

### Component 4: Drift Detection
Implements statistical drift analysis and visualization.

### Component 5: AI Risk Assessment
Includes system boundary, governance review, risk matrix, and CTO memo.

---

## Risk & Governance

Key risks:
- Data leakage  
- Retrieval contamination  
- Hallucination  
- Compliance violations  

Mitigations:
- Access control and encryption  
- Data filtering and validation  
- Monitoring and alerting  
- Policy enforcement  

---

## Reproducibility
- Fully local execution  
- Synthetic data used  
- Dependencies pinned  

---

## Technologies
- Python (pandas, numpy, matplotlib, scipy)  
- VS Code  
- Virtual environments (venv)
