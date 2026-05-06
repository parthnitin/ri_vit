# Prototypical NIDS Paper — 2026 Update Plan

> **Original Paper:** Prototypical Network for Network Intrusion Detection System using Deep Learning  
> **Authors:** Prof Dr. Sandip Shinde, Parth Nitin Mule  
> **Current Year:** 2026  
> **Status:** Final year college project → Professional-grade publication

---

## 🧭 Document Map & Reading Order

Read these documents **in order**. Each builds on the previous.

| # | Document | Read Time | What It Covers |
|---|----------|-----------|----------------|
| **00** | 📖 **This file** | 10 min | Context, reading order, your mentor strategy, timeline |
| **01** | [Literature Review & V3 Dataset](./01_Literature_Review_and_Dataset.md) | 45 min | What's changed in 2024-2026, the new NF-* V3 datasets (53 temporal features), what to cite |
| **02** | [Production NIDS & ProtoNet Advantage](./02_Production_NIDS_and_ProtoNet_Advantage.md) | 30 min | How real-world production NIDS work, where ProtoNet fits, edge deployment, optimization |
| **03** | [Full Paper Rewrite Strategy](./03_Full_Paper_Rewrite_Strategy.md) | 60 min | Complete section-by-section rewrite plan, what to keep/cut/add, new claims |
| **04** | [Implementation Plan](./04_Implementation_Plan.md) | 20 min | Week-by-week code roadmap, experiments to run, how to hit SOTA |

---

## 🎯 Your Goal

> *"I want to update my college paper to a **professional-grade publication** that could be submitted to a good journal. I want to keep the core prototypical idea, use the new V3 dataset, and make this actually SOTA or close to it."*

### What "SOTA" Means Realistically

| Claim | Where Your Paper Was | Where It Could Be |
|-------|---------------------|-------------------|
| Overall accuracy | 89% on NF-UQ-NIDS-v2 | **92-94%** on NF-UQ-NIDS-v3 |
| Minority class F1 | 0.88 (rare attacks) | **0.91-0.93** (with better encoder + episodic training) |
| Few-shot novel attacks | Not evaluated | **80-85%** on unseen classes (5-shot) |
| Cross-dataset transfer | Not evaluated | **75-80%** zero-shot on CIC-IDS2017 |
| Interpretability | Qualitative centroids | **Quantitative** (silhouette scores, inter/intra-class ratios) |
| Production readiness | Not measured | Latency, throughput, model size on edge hardware |

The honest truth: You probably won't beat a well-tuned XGBoost on **overall accuracy** (ensemble methods are still king for tabular data). But you **can** beat them on:

1. **Minority class performance** (this is the real cybersecurity win)
2. **Few-shot adaptation** to new attacks
3. **Interpretability** via centroid space
4. **Model size & deployment cost** (KB-sized centroids vs MB-sized forests)

**Your narrative:** *"We trade 1-2% overall accuracy for 15-20% better rare-attack detection, native few-shot capability, and interpretable explanations — in a model that fits in 500KB and runs on a Raspberry Pi."*

---

## 🧑‍🏫 How to Approach Your Mentor

This is a separate skill from the technical work. Here's a respectful, professional strategy:

### The Situation

- You told your mentor you'd update the paper after graduating
- A year passed with no update (this is **extremely common** — mentors expect this)
- You're now coming back with completed work

### Why This Works in Your Favor

1. **Mentors are used to students disappearing.** Showing up with done work is a positive surprise.
2. **Your mentor's name is already on the paper.** He has a natural incentive to help improve it.
3. **You're not asking for his time to "figure things out"** — you're presenting finished work and asking for mentorship on the last mile.

### Step-by-Step Approach

#### Step 1: Do the Work First (You're Here)

Complete the experiments and paper rewrite **before** contacting him. This is what separates a serious approach from another abandoned project.

#### Step 2: The Email Draft

```
Subject: Updated Prototypical NIDS paper — ready for your review

Hi Prof. Shinde,

I hope you're doing well. It's been a while since we worked on the 
Prototypical Network NIDS paper — I wanted to share an update.

Since graduating, I've been working in industry and continued developing 
the paper in my spare time. I've:

1. Reviewed the literature from 2024-2026 and updated the paper accordingly
2. Upgraded to the new NF-UQ-NIDS-V3 dataset (53 temporal features) 
3. Redesigned the encoder architecture and training pipeline
4. Added proper few-shot evaluation, cross-validation, and fair baselines
5. Written a complete production-NIDS analysis showing deployment value

The result is a substantially stronger paper that I believe could be suitable 
for [journal name] or [conference name].

I've attached the complete rewritten paper and supporting documents. I'd love 
to know if you'd be interested in co-authoring the updated version. If your 
interests or priorities have shifted, I completely understand — I'm also happy 
to proceed independently.

No pressure at all. I just wanted to share the work and see if it aligns with 
your research direction.

Best,
Parth
```

#### Step 3: What to Attach

- The rewritten paper (PDF)
- The supporting markdown documents (optional, mention they're available)
- A 1-page summary of what changed and why

#### Step 4: Handle the Response

| Response | Your Move |
|----------|-----------|
| "Great work, let's co-author and submit" | 🎉 Perfect. Ask about preferred journal, timeline |
| "Busy now, maybe later" | Thank him, ask if you can submit independently. Offer to keep him as co-author or remove — his choice |
| "I'd rather not be involved" | "Totally understand. I'll proceed independently and acknowledge your original supervision." Then publish alone |
| No response (2 weeks) | Follow up once politely. If still no response, proceed independently |

### Pro Tip

> *"Never ask for permission. Ask for collaboration on something already complete."*

You're not saying "Can I work on this?" You're saying "I've completed this — would you like to join?"

---

## 📅 Suggested Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Literature review** | 1 week | Annotated bibliography, 15-20 new refs |
| **Code: V3 dataset + new encoder** | 1 week | Working pipeline with V3 data |
| **Code: Episodic training** | 1 week | Proper few-shot training + evaluation |
| **Code: Baselines + cross-validation** | 1 week | Fair comparison table + significance tests |
| **Code: Production metrics** | 2-3 days | Latency, throughput, model size |
| **Write: Revised paper** | 1.5 weeks | Complete LaTeX draft |
| **Review & polish** | 3 days | Iterate, check claims, proofread |
| **Contact mentor** | Day 1 of Phase 8 | Send email with attached paper |
| **Target: Submit to journal** | 2 weeks after mentor response | Assuming collaborative revision |

**Total: ~6-7 weeks of focused weekend/evening work.**

---

## 📦 Document Structure

```
ri_vit/
├── paper.txt                          # Original LaTeX paper
├── methodology_review.md              # Previous review
│
├── update_plan/                       # ← NEW: All planning docs
│   ├── 00_READ_THIS_FIRST.md          # This file
│   ├── 01_Literature_Review_and_Dataset.md
│   ├── 02_Production_NIDS_and_ProtoNet_Advantage.md
│   ├── 03_Full_Paper_Rewrite_Strategy.md
│   └── 04_Implementation_Plan.md
│
├── rewritten_paper/                   # ← FUTURE: New LaTeX files
│   ├── paper_v2.tex
│   └── figures/
│
└── code/                              # ← FUTURE: New experiments
    ├── protonet_v2.py
    ├── baselines.py
    └── evaluation.py
```

---

## ❓ FAQ

### "Should I keep the same title?"

**Yes, with a minor update:**
- Old: *"Prototypical Network for Network Intrusion Detection System using Deep Learning"*
- New: *"Prototypical Networks for Network Intrusion Detection: Few-Shot Learning with Temporal NetFlow Features"*

More descriptive, still clearly about ProtoNet, highlights the new contribution (temporal features).

### "What if my mentor says no?"

Publish independently. The original paper was your work too. Acknowledge his supervision in the acknowledgements. Move forward.

### "Is this good enough for a journal?"

With the upgrades described in these documents, yes. Target these venues:

| Venue | Reach | Difficulty | Best Fit |
|-------|-------|------------|----------|
| **Computers & Security (Elsevier)** | Top NIDS journal | High | If you nail the production/real-world angle |
| **IEEE Access** | Broad, open access | Medium | Good target — accepts well-executed work |
| **Journal of Information Security and Applications** | Good NIDS fit | Medium | Strong methodology paper |
| **Expert Systems with Applications** | Applied ML | Medium-High | If you emphasize deployment metrics |

### "Should I switch to Python-first or stay with the original framework?"

Python (PyTorch). Your original paper doesn't show code, so no lock-in. PyTorch is the standard for research.

---

**Next:** Read [01_Literature_Review_and_Dataset.md](./01_Literature_Review_and_Dataset.md) to understand what's changed in the last 2 years and how to leverage the V3 dataset.
