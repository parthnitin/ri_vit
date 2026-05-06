# Comprehensive Literature Survey: ProtoNet for NIDS (2024-2026)

> **Purpose:** Your complete literature survey methodology and paper database.  
> **How to use:** Read this document. Then open each paper link. Skim. Track in your spreadsheet.  
> **Total papers found:** 50+ from live arXiv search (May 2026)

---

## Part 1: How to Conduct Your Literature Survey

### The Method (3 Passes)

```
Pass 1 (30 sec/paper): Read title + abstract only
    → Keep if relevant. Trash if not. Maybe-save to a list.
    
Pass 2 (10 min/paper): Read intro last paragraph + results table + conclusion
    → Note: dataset, method, accuracy, what they did NOT do (gap)
    
Pass 3 (1 hour/paper): Read the full paper, take structured notes
    → For papers most directly relevant to your work
```

### Your Tracking Template

Create this file: `literature_tracker.md`

```markdown
## [Paper #] [Title] ([Year])
- **Link:** [URL]
- **Category:** [ProtoNet direct / NIDS baseline / Temporal / Production / Encoder / XAI / Other]
- **Pass:** [1 / 2 / 3]
- **Dataset:** [e.g., NF-UNSW-NB15-v3, CIC-IDS2017, NSL-KDD]
- **Method:** [e.g., ProtoNet + metric fusion]
- **Key result:** [e.g., 91% binary accuracy]
- **What I can cite them for:** [e.g., ProtoNet works for NIDS]
- **Gap (my contribution):** [e.g., No temporal features, no few-shot eval]
- **Notes:** [anything worth remembering]
```

### Search Queries to Run on Google Scholar

Copy-paste these into [scholar.google.com](https://scholar.google.com):

```
1. "prototypical network" "intrusion detection" 2024 2025 2026
2. "few-shot learning" "network intrusion" 2024 2025
3. "temporal features" NetFlow intrusion detection 2025
4. "metric learning" "network intrusion detection"
5. "NF-UQ-NIDS" OR "NetFlow V3" intrusion
6. "tabular deep learning" intrusion detection
7. "self-supervised" network traffic intrusion
8. "continual learning" intrusion detection
9. "explainable AI" network intrusion detection
10. "lightweight" "edge" "intrusion detection" deployment
11. "transformer" "network intrusion detection" 2024 2025
12. "graph neural network" "network intrusion detection"
```

On **arXiv** ([arxiv.org](https://arxiv.org)):
```
1. prototypical network intrusion detection
2. few-shot network intrusion detection
3. NetFlow V3 temporal features
```

---

## Part 2: Paper Database (50 Papers Found)

### Category A: ProtoNet + Few-Shot NIDS (Directly Relevant — READ FIRST)

| # | Paper | Year | Link | Key Point | Gap Your Paper Fills |
|---|-------|------|------|-----------|---------------------|
| **A1** | **Snell et al.** — *Prototypical Networks for Few-shot Learning* ⭐ | 2017 | [arxiv.org/abs/1703.05175](https://arxiv.org/abs/1703.05175) | **The original ProtoNet paper** — defines episodic training, centroid-based classification | Foundation you build on |
| **A2** | **Martinez-Lopez et al.** — *Learning in Multiple Spaces: Few-Shot Network Attack Detection with Metric-Fused Prototypical Networks* ⭐ | 2024 | [arxiv.org/abs/2501.00050](https://arxiv.org/abs/2501.00050) | Proposes **Multi-Space Prototypical Learning (MSPL)** — multiple embedding spaces fused with learned metrics | No temporal features, no production metrics, no cross-dataset eval |
| **A3** | **Niknami et al.** — *PTN-IDS: Prototypical Network for Few-shot Detection in IDS* ⭐ | 2024 | IEEE LCN 2024 (search) | ProtoNet for NIDS, 91% binary accuracy on CIC-IDS 2017/2018 | Binary only, no temporal features, no few-shot evaluation on novel classes |
| **A4** | **Sun et al.** — *Few-Shot NIDS based on Prototypical Capsule Network with Attention* | 2023 | PLOS ONE (search) | ProtoNet + capsule network + attention, 95.22% on NSL-KDD | Uses outdated NSL-KDD dataset |
| **A5** | **Ben Atitallah et al.** — *Strengthening NIDS in IoT with Self-Supervised Learning and Few Shot Learning* | 2024 | [arxiv.org/abs/2406.02636](https://arxiv.org/abs/2406.02636) | Self-supervised pretraining → few-shot ProtoNet for IoT NIDS | IoT-specific, not temporal NetFlow |
| **A6** | **Your original paper** — *Prototypical Network for NIDS using Deep Learning* | 2024 | Local copy | Your 89% F1 on NF-UQ-NIDS-v2, centroid visualization | You're extending it |

### Category B: NetFlow Datasets + Temporal Features (High Relevance)

| # | Paper | Year | Link | Key Point | Paper Use |
|---|-------|------|------|-----------|-----------|
| **B1** | **Luay et al.** — *Temporal Analysis of NetFlow Datasets for NIDS* ⭐ | 2025 | [arxiv.org/abs/2503.04404](https://arxiv.org/abs/2503.04404) | **The V3 dataset paper** — 10 temporal features, time-frequency analysis of attacks | Your primary dataset source |
| **B2** | **Sarhan et al.** — *Towards a Standard Feature Set for NIDS Datasets* ⭐ | 2022 | [arxiv.org/abs/2011.09144](https://arxiv.org/abs/2011.09144) | **Original V1/V2 dataset paper** — unified 43-feature NetFlow format | Background, V1/V2 baseline comparisons |
| **B3** | **El Mahdaouy et al.** — *Deep Learning for Contextualized NetFlow-Based NIDS: Methods, Data, Evaluation and Deployment* ⭐ | 2026 | [arxiv.org/abs/2602.05594](https://arxiv.org/abs/2602.05594) | **Comprehensive 2026 survey** — covers full pipeline from data to deployment | Use for Lit Review structure + production claims |

### Category C: NIDS Baselines (For Your Comparison Table)

| # | Paper | Year | Link | Key Point | What to Cite |
|---|-------|------|------|-----------|-------------|
| **C1** | **Hnamte et al.** — *Two-Stage Deep Learning: LSTM-AE* | 2023 | IEEE Access (search) | LSTM + Autoencoder, 99.1% on CICIDS2017 | High accuracy but single dataset |
| **C2** | **Cao et al.** — *CNN + GRU with Hybrid Sampling* | 2022 | Applied Sciences (search) | CNN-GRU, ADASYN+RENN, 99.69% on NSL-KDD | Dramatic dataset-dependent drop (86% on UNSW-NB15) |
| **C3** | **Ravi et al.** — *Recurrent DL Ensemble with KPCA* | 2022 | Computers & Electrical Engineering (search) | Ensemble meta-classifier, 99% on 4 datasets | Cross-dataset evaluation approach |
| **C4** | **Du et al.** — *NIDS-CNNLSTM* | 2023 | IEEE Access (search) | CNN-LSTM on UNSW-NB15, 82.9% acc, 49.5% F1 | Low recall example |
| **C5** | **Halbouni et al.** — *CNN-LSTM Hybrid* | 2022 | IEEE Access (search) | 99.64% on CIC-IDS2017, 94.53% on UNSW-NB15 | Dataset-dependent variance |
| **C6** | **Said et al.** — *CNN-BiLSTM for SDN* | 2023 | IEEE Access (search) | CNN-BiLSTM, 97.12% on NSL-KDD | Binary vs multiclass gap |
| **C7** | **Zia et al.** — *Zero-Touch Network Security (ZTNS)* | 2024 | IEEE Access (search) | 1D-CNN on CIC-IDS, 99.8% | High accuracy, no few-shot |

### Category D: Transformer + Attention Encoders (Encoder Upgrade Ideas)

| # | Paper | Year | Link | Key Point | For Your Encoder |
|---|-------|------|------|-----------|-----------------|
| **D1** | **Butt et al.** — *Evaluating Tabular Representation Learning for NIDS* ⭐ | 2026 | [arxiv.org/abs/2605.02519](https://arxiv.org/abs/2605.02519) | **Very recent (May 2026)!** Compares tabular encoders for NIDS | Guide for which encoder to use |
| **D2** | **She** — *PPO-optimized Tabular Transformer for IIoT Intrusion Detection* | 2025 | [arxiv.org/abs/2505.18234](https://arxiv.org/abs/2505.18234) | Tabular Transformer + PPO optimization | Transformer encoder reference |
| **D3** | **Zhang et al.** — *Transformer-BiGRU with Data Augmentation* | 2025 | [arxiv.org/abs/2509.04925](https://arxiv.org/abs/2509.04925) | Transformer-BiGRU hybrid for NIDS | Temporal + attention hybrid |
| **D4** | **Joshi & Gurusamy** — *Time Series NIDS using MTF-Aided Transformer* | 2025 | [arxiv.org/abs/2508.16035](https://arxiv.org/abs/2508.16035) | Markov Transition Field + Transformer | Temporal feature encoding |
| **D5** | **Koukoulis et al.** — *Self-Supervised Transformer Contrastive Learning for IDS* | 2025 | [arxiv.org/abs/2505.08816](https://arxiv.org/abs/2505.08816) | Contrastive SSL + Transformer | Self-supervised pretraining for encoder |

### Category E: Self-Supervised + Representation Learning (Future Scope)

| # | Paper | Year | Link | Key Point |
|---|-------|------|------|-----------|
| **E1** | **Guerra et al.** — *Self-Supervised Learning of Graph Representations for NIDS* | 2025 | [arxiv.org/abs/2509.16625](https://arxiv.org/abs/2509.16625) | Graph SSL for NIDS |
| **E2** | **Van Langendonck et al.** — *Towards a Graph-based Foundation Model for Network Traffic* | 2024 | [arxiv.org/abs/2409.08111](https://arxiv.org/abs/2409.08111) | Foundation model idea for network traffic |
| **E3** | **Nakıp & Gelenbe** — *Online Self-Supervised Deep Learning for IDS* | 2023 | [arxiv.org/abs/2306.13030](https://arxiv.org/abs/2306.13030) | Online SSL |
| **E4** | **Caville et al.** — *Anomal-E: Self-Supervised GNN for NIDS* | 2022 | [arxiv.org/abs/2207.06819](https://arxiv.org/abs/2207.06819) | SSL + GNN |
| **E5** | **Xu et al.** — *Self-Supervised Learning for Network Flows with GNN* | 2024 | [arxiv.org/abs/2403.01501](https://arxiv.org/abs/2403.01501) | SSL on NetFlow data |
| **E6** | **Menssouri & Amhoud** — *PrivFly: Privacy-Preserving SSL for Rare Attack Detection* | 2026 | [arxiv.org/abs/2601.13003](https://arxiv.org/abs/2601.13003) | SSL for rare attacks |

### Category F: Continual Learning + Novelty Detection (ProtoNet's Incremental Advantage)

| # | Paper | Year | Link | Key Point |
|---|-------|------|------|-----------|
| **F1** | **Fuhrman et al.** — *ACORN-IDS: Adaptive Continual Novelty Detection for IDS* ⭐ | 2026 | [arxiv.org/abs/2602.07291](https://arxiv.org/abs/2602.07291) | Continual novelty detection — compares well with ProtoNet's approach |
| **F2** | **Fuhrman et al.** — *CND-IDS: Continual Novelty Detection for IDS* | 2025 | [arxiv.org/abs/2502.14094](https://arxiv.org/abs/2502.14094) | Earlier version of F1 |
| **F3** | **Banerjee et al.** — *Quantifying Catastrophic Forgetting in IoT IDS* | 2026 | [arxiv.org/abs/2603.00363](https://arxiv.org/abs/2603.00363) | **Show that ProtoNet doesn't suffer from this** |
| **F4** | **Zhang et al.** — *Continual Learning with Strategic Selection and Forgetting for NIDS* | 2024 | [arxiv.org/abs/2412.16264](https://arxiv.org/abs/2412.16264) | Continual learning approach |
| **F5** | **Li et al.** — *CITADEL: Continual Anomaly Detection for IoT IDS* | 2025 | [arxiv.org/abs/2508.19450](https://arxiv.org/abs/2508.19450) | Anomaly detection with continual learning |

### Category G: Explainable AI for NIDS (Your Centroid Interpretation Advantage)

| # | Paper | Year | Link | Key Point |
|---|-------|------|------|-----------|
| **G1** | **Kalakoti et al.** — *Evaluating XAI for DL-based NIDS Alert Classification* | 2025 | [arxiv.org/abs/2506.07882](https://arxiv.org/abs/2506.07882) | XAI evaluation for NIDS |
| **G2** | **Kumar et al.** — *ExpIDS: Drift-adaptable NIDS with Improved Explainability* | 2025 | [arxiv.org/abs/2509.20767](https://arxiv.org/abs/2509.20767) | Explainability + drift adaptation |
| **G3** | **Galwaduge & Samarabandu** — *Tabular Diffusion-based Counterfactual Explanations for NIDS* | 2025 | [arxiv.org/abs/2507.17161](https://arxiv.org/abs/2507.17161) | Counterfactual explanations for tabular NIDS |
| **G4** | **Sheikhi et al.** — *ExAI5G: Logic-based XAI for IDS in 5G* | 2026 | [arxiv.org/abs/2604.18052](https://arxiv.org/abs/2604.18052) | XAI for 5G NIDS |
| **G5** | **Yang et al.** — *Large Language Models for NIDS: Foundations and Future* | 2025 | [arxiv.org/abs/2507.04752](https://arxiv.org/abs/2507.04752) | LLM-based NIDS survey |

### Category H: Production + Edge Deployment (Your Deployment Metrics Section)

| # | Paper | Year | Link | Key Point |
|---|-------|------|------|-----------|
| **H1** | **Alikhani** — *DKD-KAN: Lightweight Knowledge-Distilled KAN for IDS* | 2026 | [arxiv.org/abs/2603.03486](https://arxiv.org/abs/2603.03486) | Lightweight model distillation |
| **H2** | **Kuznetsov** — *LUT-Compiled KAN for Lightweight DoS Detection on IoT Edge* | 2026 | [arxiv.org/abs/2601.08044](https://arxiv.org/abs/2601.08044) | Extremely lightweight edge deployment |
| **H3** | **Benaddi et al.** — *Lightweight IDS via SHAP-Guided Feature Pruning + Knowledge Distillation* | 2025 | [arxiv.org/abs/2512.19488](https://arxiv.org/abs/2512.19488) | Feature pruning + distillation |
| **H4** | **Rahmati** — *Explainable and Lightweight AI for Real-Time Threat Hunting* | 2025 | [arxiv.org/abs/2504.16118](https://arxiv.org/abs/2504.16118) | Real-time edge deployment |

### Category I: Imbalanced Learning for NIDS (Your Minority Class Advantage)

| # | Paper | Year | Link | Key Point |
|---|-------|------|------|-----------|
| **I1** | **Zeng** — *C²BNVAE: Dual-Conditional Deep Generation for NIDS Balancing* | 2025 | [arxiv.org/abs/2506.05844](https://arxiv.org/abs/2506.05844) | Synthetic data generation for class balance |
| **I2** | **Zeng** — *CSAGC-IDS: Dual-Module DL for Complex & Imbalanced Data* | 2025 | [arxiv.org/abs/2505.14027](https://arxiv.org/abs/2505.14027) | DL for imbalanced NIDS |
| **I3** | **Aravind et al.** — *Diffusion-Driven Synthetic Tabular Data for DoS/DDoS* | 2026 | [arxiv.org/abs/2601.13197](https://arxiv.org/abs/2601.13197) | Diffusion models for synthetic attack data |
| **I4** | **Kara et al.** — *HybridGuard: Minority-Class Intrusion Detection in Edge-of-Things* | 2025 | [arxiv.org/abs/2511.07793](https://arxiv.org/abs/2511.07793) | Minority class focus |
| **I5** | **Baidar et al.** — *Hybrid DL-FL Powered IDS for IoT/5G Edge* | 2025 | [arxiv.org/abs/2509.15555](https://arxiv.org/abs/2509.15555) | Federated learning + IDS |

### Category J: GNN-based NIDS (Alternative Architecture for Future Scope)

| # | Paper | Year | Link | Key Point |
|---|-------|------|------|-----------|
| **J1** | **Zhan et al.** — *REAL-IoT: GNN Robustness under Practical Adversarial Attack* | 2025 | [arxiv.org/abs/2507.10836](https://arxiv.org/abs/2507.10836) | GNN robustness |
| **J2** | **Masukawa et al.** — *PacketCLIP: Multi-Modal Embedding of Network Traffic and Language* | 2025 | [arxiv.org/abs/2503.03747](https://arxiv.org/abs/2503.03747) | CLIP-style multimodal traffic analysis |
| **J3** | **Jahin et al.** — *CAGN-GAT Fusion: Contrastive Attentive GNN for NIDS* | 2025 | [arxiv.org/abs/2503.00961](https://arxiv.org/abs/2503.00961) | Contrastive + GNN |
| **J4** | **Farrukh et al.** — *XG-NID: Heterogeneous GNN + LLM for NIDS* | 2024 | [arxiv.org/abs/2408.16021](https://arxiv.org/abs/2408.16021) | GNN + LLM |
| **J5** | **Van Langendonck et al.** — *PPT-GNN: Pre-Trained Spatio-Temporal GNN for Network Security* | 2024 | [arxiv.org/abs/2406.13365](https://arxiv.org/abs/2406.13365) | Spatio-temporal GNN pretraining |

### Category K: Surveys & Meta-Reviews (For Lit Review Structure)

| # | Paper | Year | Link | Key Point |
|---|-------|------|------|-----------|
| **K1** | **Tripathy & Behera** — *Review of Datasets for ML-based IDS* | 2025 | [arxiv.org/abs/2506.02438](https://arxiv.org/abs/2506.02438) | Dataset comparison survey |
| **K2** | **Yang et al.** — *Large Language Models for NIDS: Foundations, Implementations, and Future Directions* | 2025 | [arxiv.org/abs/2507.04752](https://arxiv.org/abs/2507.04752) | LLM + NIDS survey |
| **K3** | **Chou & Jiang** — *Survey on Data-Driven NIDS* (your original ref [20]) | 2021 | ACM Computing Surveys | Already in your bibliography |

---

## Part 3: Reading Priority

### Tier 1: Must Read (Pass 3 — Full Read, 1 hour each)

These directly define your contribution. Read first.

| Order | Paper | Why |
|-------|-------|-----|
| 1 | **A1: Snell et al. (2017)** — Original ProtoNet | Foundation of your entire approach |
| 2 | **B1: Luay et al. (2025)** — V3 Dataset | Defines your dataset, gives baselines to beat |
| 3 | **A2: Martinez-Lopez et al. (2024)** — MSPL | Most similar approach — you need to differentiate |
| 4 | **B3: El Mahdaouy et al. (2026)** — NetFlow Survey | Production deployment context |

### Tier 2: Important Context (Pass 2 — Skim, 10 min each)

| Order | Paper | Why |
|-------|-------|-----|
| 5 | **A3: Niknami et al. (2024)** — PTN-IDS | Direct ProtoNet competitor |
| 6 | **D1: Butt et al. (2026)** — Tabular Encoders for NIDS | Encoder architecture guide |
| 7 | **F1: Fuhrman et al. (2026)** — ACORN-IDS | Continual novelty detection (similar to ProtoNet's incremental class addition) |
| 8 | **F3: Banerjee et al. (2026)** — Catastrophic Forgetting | Use to show ProtoNet's advantage (no forgetting) |
| 9 | **C1-C7** — Baseline models (pick 3) | Comparison table |

### Tier 3: Broader Context (Pass 1 — Abstract only, 30 sec each)

Read all remaining papers' abstracts. Decide which 5-10 to keep. The rest are optional.

### Target: 15-20 Papers at Pass 2/3 Level

This is enough for a solid literature review section. Don't try to read all 50 deeply.

---

## Part 4: What You Can Do with ProtoNet (It's Flexible)

You asked about keeping the ProtoNet base but doing different things. Here's a menu:

### Level 1: Keep Everything, Just Upgrade (Your Current Plan)

```
ProtoNet base:     Same (centroid-based classification)
Encoder:           Upgrade (residual MLP or FT-Transformer)
Dataset:           Upgrade (V2 → V3, temporal features)
Training:          Fix (standard batch → episodic training)
Evaluation:        Add (few-shot, cross-validation, baselines)
Production:        Add (latency, size, edge deployment)
```

**Risk:** Low. **Novelty:** Medium. **Effort:** 6-8 weeks.

### Level 2: Change the Encoder Dramatically

```
ProtoNet base:     Same
Encoder:           GNN (flow graph) or Transformer or TabNet
Dataset:           V3
Training:          Episodic + self-supervised pretraining
```

**Risk:** Medium (GNN needs graph construction). **Novelty:** High. **Effort:** 8-12 weeks.

### Level 3: Multi-Prototype or Hierarchical ProtoNet

```
ProtoNet base:     Modified — multiple prototypes per class instead of one
Encoder:           Residual MLP
Dataset:           V3
Idea:              Some attacks have multiple sub-behaviors (DoS has different types)
                   → Learn 3 centroids per class
                   → Or hierarchical: Benign → Attack type → Sub-type
```

**Risk:** Medium. **Novelty:** High (nobody has done this for NIDS). **Effort:** 8-10 weeks.

### Level 4: ProtoNet + Active Learning

```
ProtoNet base:     Same
Add:               Uncertainty sampling via distance to nearest centroid
                   → Flag samples far from all centroids → ask human to label
                   → Add as new centroid (few-shot expansion)
```

**Risk:** Low-Medium. **Novelty:** High. **Effort:** 6-8 weeks. **Great for production narrative.**

### Level 5: ProtoNet + Continual Learning

```
ProtoNet base:     Same
Add:               Online centroid updating (weighted moving average)
                   → Old attacks fade, new attacks appear
                   → No full retraining needed
                   → Compare to: Banerjee et al. (F3) catastrophic forgetting
```

**Risk:** Low. **Novelty:** High. **Effort:** 4-6 weeks. **Directly publishable.**

### Level 6: Transductive ProtoNet for NIDS

```
ProtoNet base:     Modified — query set used to refine centroids during inference
Encoder:           Residual MLP
Dataset:           V3
Idea:              During inference, refine centroids using unlabeled test samples
                   → Better for distribution shift scenarios
```

**Risk:** Medium. **Novelty:** High (no transductive ProtoNet for NIDS exists). **Effort:** 6-8 weeks.

### My Recommendation: Level 1 + Level 5

Do the upgrades (Level 1) but also add **online centroid updating** (Level 5) as a main contribution. This gives you:

1. **Incremental new class addition** (5-shot → new centroid → deployed in seconds)
2. **Catastrophic forgetting immunity** (centroids are additive, not weight-updating)
3. **Deployment-friendly** (no GPU needed for updates)

This directly addresses a gap that **nobody in the current literature fills** — and it's simple to implement (about 50 lines of code).

---

## Part 5: Next Steps — Your Action Plan

| Step | What | Time |
|------|------|------|
| **1** | Create `literature_tracker.md`, paste the template | 5 min |
| **2** | Open **Tier 1 papers** (A1, B1, A2, B3) — read abstracts | 5 min |
| **3** | Skim **Tier 1 papers** — Pass 2 (intro + results + conclusion) | 40 min |
| **4** | Open **Tier 2 papers** (A3, D1, F1, F3) — read abstracts | 5 min |
| **5** | Skim **Tier 2 papers** — Pass 2 | 30 min |
| **6** | Deep read **Tier 1** papers — Pass 3 (choose 2 of 4) | 2 hours |
| **7** | Start filling your tracker with structured notes | 1 hour |
| **8** | Decide: Level 1 only or Level 1 + Level 5? | 30 min |
| **9** | Decide on final encoder architecture (residual MLP vs FT-Transformer) | 30 min |
| **10** | **Start coding** (Phase 0 from Implementation Plan) | — |

**Total reading time:** ~4 hours spread across 3-4 sessions. Do it before you write a single line of code.

---

## Part 6: How Your Literature Review Section Maps to These Papers

| Lit Review Subsection | Papers to Cite |
|-----------------------|----------------|
| **1. Traditional ML for NIDS** | C1-C7, K1 |
| **2. Deep Learning for NIDS** | C1-C7, D3, D4 |
| **3. Transformers & Modern Encoders for NIDS** | D1, D2, D5 |
| **4. Few-Shot Learning & Prototypical Networks** | A1, A2, A3, A4, A5, **A6** (your original) |
| **5. Temporal Features & NetFlow Datasets** | B1, B2, B3 |
| **6. Explainable AI & Interpretability** | G1-G5 |
| **7. Production & Edge NIDS Deployment** | H1-H4, B3 |
| **8. Continual Learning & Novelty Detection** | F1-F5 |
| **9. Self-Supervised & Representation Learning** | E1-E6 |
| **10. Research Gap & Your Contribution** | All of the above → point to what they're missing |
