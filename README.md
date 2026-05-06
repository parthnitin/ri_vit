# Prototypical Network for Network Intrusion Detection System using Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **Prototypical Network (ProtoNet)**-based approach for Network Intrusion Detection, using Euclidean distance-based classification to learned class centroids. This repository contains the original paper, a comprehensive methodology review, and a complete 2026 update plan with the new NF-UQ-NIDS-V3 temporal NetFlow datasets.

## 📄 Documents

| Document | Description |
|----------|-------------|
| [`base_documents/paper.txt`](base_documents/paper.txt) | Original 2024 paper (LaTeX source) |
| [`base_documents/methodology_review.md`](base_documents/methodology_review.md) | Comprehensive methodology critique (15 key issues identified) |
| [`update_plan/00_READ_THIS_FIRST.md`](update_plan/00_READ_THIS_FIRST.md) | **Start here** — context, reading order, mentor strategy |
| [`update_plan/01_Literature_Review_and_Dataset.md`](update_plan/01_Literature_Review_and_Dataset.md) | 2024–2026 literature survey + NF-UQ-NIDS-V3 dataset deep-dive |
| [`update_plan/02_Production_NIDS_and_ProtoNet_Advantage.md`](update_plan/02_Production_NIDS_and_ProtoNet_Advantage.md) | How production NIDS work + ProtoNet's deployment advantages |
| [`update_plan/03_Full_Paper_Rewrite_Strategy.md`](update_plan/03_Full_Paper_Rewrite_Strategy.md) | Complete section-by-section paper rewrite plan |
| [`update_plan/04_Implementation_Plan.md`](update_plan/04_Implementation_Plan.md) | Week-by-week code roadmap (7 weeks to SOTA) |

## 🧭 Reading Order

```
00_READ_THIS_FIRST.md  (context & strategy)
        ↓
01_Literature_Review_and_Dataset.md  (what's changed)
        ↓
02_Production_NIDS_and_ProtoNet_Advantage.md  (real-world value)
        ↓
03_Full_Paper_Rewrite_Strategy.md  (the new paper)
        ↓
04_Implementation_Plan.md  (how to build it)
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
