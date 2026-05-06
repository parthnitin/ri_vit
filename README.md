# Prototypical Network for Network Intrusion Detection System using Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **Prototypical Network (ProtoNet)**-based approach for Network Intrusion Detection, using Euclidean distance-based classification to learned class centroids. This repository contains the original paper, a comprehensive methodology review, and a complete 2026 update plan with the new NF-UQ-NIDS-V3 temporal NetFlow datasets.

## 📄 Documents

| Document | Description |
|----------|-------------|
| [`base_documents/paper.txt`](base_documents/paper.txt) | Original 2024 paper (LaTeX source) |
| [`base_documents/methodology_review.md`](base_documents/methodology_review.md) | Comprehensive methodology critique (15 key issues identified) |
| Document | Type | Description |
|----------|------|-------------|
| [`base_documents/paper.txt`](base_documents/paper.txt) | Reference | Original 2024 paper (LaTeX source) |
| [`base_documents/methodology_review.md`](base_documents/methodology_review.md) | Reference | Comprehensive methodology critique (15 key issues identified) |
| | | |
| [`update_plan/00_READ_THIS_FIRST.md`](update_plan/00_READ_THIS_FIRST.md) | **Read first** | Context, mentor strategy, timeline |
| [`update_plan/06_Literature_Survey_Methodology.md`](update_plan/06_Literature_Survey_Methodology.md) | **Read second** | **Complete literature survey methodology** — how to find papers, 50 papers found across 10 categories with links, tracking template, 6 levels of ProtoNet innovation you can pursue |
| [`update_plan/01_Literature_Review_and_Dataset.md`](update_plan/01_Literature_Review_and_Dataset.md) | Read third | 2024–2026 literature analysis + NF-UQ-NIDS-V3 dataset deep-dive (read AFTER your own survey to cross-check) |
| [`update_plan/02_Production_NIDS_and_ProtoNet_Advantage.md`](update_plan/02_Production_NIDS_and_ProtoNet_Advantage.md) | Read fourth | How production NIDS work + ProtoNet's deployment advantages |
| [`update_plan/03_Full_Paper_Rewrite_Strategy.md`](update_plan/03_Full_Paper_Rewrite_Strategy.md) | Read fifth | Complete section-by-section paper rewrite plan |
| [`update_plan/04_Implementation_Plan.md`](update_plan/04_Implementation_Plan.md) | Read sixth | Week-by-week code roadmap (7 weeks to SOTA) |
| [`update_plan/05_Temporal_Features_Changes.md`](update_plan/05_Temporal_Features_Changes.md) | **Code reference** | V3 temporal features: NaN handling, 53 features, encoder input dim. Refer to **while coding**, not before |

## 🧭 Reading Order (Do NOT read linearly — follow this phased approach)

### Phase 0: Orientation (read in one sitting, 30 min total)

```
00_READ_THIS_FIRST.md   → Context, mentor strategy, timeline
```

### Phase 1: Do Your Own Literature Survey (1-2 weeks of reading papers)

```
06_Literature_Survey_Methodology.md   → The METHOD for finding & tracking papers
                                          Then go read papers yourself:
                                          1. Open the links in this doc
                                          2. Read abstracts (Pass 1)
                                          3. Skim key papers (Pass 2)
                                          4. Deep read 2-3 most relevant (Pass 3)
                                          5. Fill your literature tracker
```

### Phase 2: Compare Your Findings With My Analysis (read after your survey, 1-2 hours)

```
01_Literature_Review_and_Dataset.md   → Check: did I miss anything?
02_Production_NIDS_and_ProtoNet_Advantage.md → Real-world NIDS context
```

### Phase 3: Plan Your Approach (read before coding, 2 hours)

```
03_Full_Paper_Rewrite_Strategy.md   → What the new paper will look like
04_Implementation_Plan.md            → Week-by-week code roadmap
```

### Phase 4: Code Reference (read just before each coding session)

```
05_Temporal_Features_Changes.md     → V3 changes: NaN handling, 53 features, encoder input dim
                                      (Refer to this WHILE coding, not before)
```

## 🎯 Project Goal

Upgrade the original 2024 college paper to a **professional-grade publication** suitable for journals like *Computers & Security* or *IEEE Access*, targeting SOTA-level results on the new NF-UQ-NIDS-V3 dataset with **53 temporal NetFlow features**.

### Key Improvements in Progress

- ✅ Switch from V2 → **V3 temporal dataset** (53 features)
- ✅ Deeper **residual encoder** architecture
- ✅ Proper **episodic training** for genuine few-shot capability
- ✅ **Cross-validation** + statistical significance
- ✅ Fair **baseline comparisons** on same splits
- ✅ **Production metrics** (latency, throughput, model size)
- ✅ **Few-shot novel attack evaluation** (held-out classes)

## 📊 Original Paper Summary

- **Model:** Prototypical Network (ProtoNet) with Euclidean distance
- **Dataset:** NF-UQ-NIDS-V2 (first 2M rows of ~76M)
- **Performance:** 89% accuracy, 0.89 macro F1
- **Key Strength:** Excellent minority class detection (F1 up to 1.00 for rare attacks)
- **Key Weakness:** Lower recall on benign traffic (0.59), no few-shot evaluation

## 👥 Authors

- **Prof Dr. Sandip Shinde** — Vishwakarma Institute of Technology, Pune
- **Parth Nitin Mule** — Vishwakarma Institute of Technology, Pune

## 📝 License

This project is open for academic use. See [LICENSE](LICENSE) for details.

---

*Status: Pre-publication. Manuscript update in progress (2026).*
