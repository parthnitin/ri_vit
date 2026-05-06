# Literature Review & V3 Dataset Analysis

> **Purpose:** Understand what's changed in NIDS research (2024-2026), deep-dive into the new NF-* V3 datasets, and identify gaps your paper can fill.  
> **Read after:** `00_READ_THIS_FIRST.md`  
> **Read before:** `02_Production_NIDS_and_ProtoNet_Advantage.md`

---

## 1. The Landscape in 2026: What's Changed Since Your Original Paper

### 1.1 The "ProtoNet for NIDS" Niche Has Grown

Back when you wrote your paper (2024), ProtoNet for NIDS was novel. Now, several papers have followed:

| Paper | Year | Dataset | Accuracy | Key Difference from Yours |
|-------|------|---------|----------|--------------------------|
| Niknami et al. (PTN-IDS) | 2024 | CIC-IDS 2017/2018 | 91.02% | Binary classification only (attack/benign) |
| Sun et al. | 2023 | NSL-KDD | 95.22% | Used capsule networks + attention |
| **Your original paper** | **2024** | **NF-UQ-NIDS-v2** | **89.0%** | **Multiclass (15 classes), centroid viz** |

**The gap that still exists:** None of these papers do all of:
1. ✅ **Proper episodic training** (most skip this — yours currently does too)
2. ✅ **Multiclass with 15+ attack types** (most do binary or 5-7 classes)
3. ✅ **Temporal features** (nobody has used V3 datasets yet for ProtoNet)
4. ✅ **Production deployment metrics** (latency, size, throughput)
5. ✅ **Cross-dataset evaluation** (nobody tests ProtoNet across multiple datasets)
6. ✅ **Statistical significance** (most report single numbers)

Your paper can be the **first** to do all six.

### 1.2 The End of "Just Stack Layers"

In 2024-2026, NIDS research has shifted:
- **2022-2023:** "Let's stack CNN + LSTM + Attention and get 99%" (on easy datasets)
- **2024-2026:** "How do we actually deploy this? What about rare attacks? What about new attacks?"

This shift helps you. Your ProtoNet approach directly addresses the hard problems the field is now focusing on.

### 1.3 Papers You Must Cite (to show you know the field)

#### Must-cite (directly relevant to your approach):

1. **Luay et al. (2025)** — *"Temporal Analysis of NetFlow Datasets for NIDS"* — This is the V3 dataset paper. You need to cite this, explain V3's temporal features, and use it.
2. **Niknami et al. (2024)** — *"PTN-IDS: Prototypical Network Solution for Few-shot Detection in IDS"* — Most directly comparable to your work. Cite and contrast.
3. **Sun et al. (2023)** — *"Few-Shot Network Intrusion Detection Based on Prototypical Capsule Network with Attention"* — Another ProtoNet variant. Cite and contrast.

#### Important context (broader NIDS advances):

4. **Hnamte et al. (2023)** — LSTM-AE two-stage model — shows the autoencoder approach
5. **Cao et al. (2024)** — CNN-GRU with hybrid sampling — best results on NSL-KDD (99.69%)
6. **Sarhan et al. (2022)** — *"Towards a Standard Feature Set for NIDS Datasets"* — The original V1/V2 paper that created the unified feature set

#### Cutting-edge (2024-2026) that positions your work:

7. **Any 2025 paper on foundation models for cybersecurity** (search: "network intrusion foundation model 2025")
8. **Any 2025 paper on temporal feature engineering for NIDS** (the V3 dataset opens this area)
9. **Any 2025 paper on few-shot/zero-shot cyber threat detection**

### 1.4 What's Actually "SOTA" in 2026?

Be honest about this — it makes your paper stronger:

| Approach | Max Accuracy Reported | Realistic Expectation | Note |
|----------|---------------------|----------------------|------|
| Gradient Boosting (XGBoost, LightGBM) | 93-96% | Still the tabular data king | Unbeatable on overall accuracy |
| Deep ensemble (CNN+LSTM+Attention) | 95-99% | Usually overfits to one dataset | Falls apart on cross-dataset tests |
| Transformer (TabTransformer, FT-Transformer) | 92-95% | Good but needs lots of data | Training cost is high |
| **ProtoNet (your approach)** | **89-93%** | **Competitive with unique advantages** | **Best for rare attacks, few-shot, interpretability** |

**Your positioning:** *"ProtoNet is not the overall accuracy leader. It is the leader in the three things that matter most for real NIDS: detecting rare attacks, adapting to novel attacks, and providing interpretable explanations."*

---

## 2. The V3 Dataset Deep-Dive

### 2.1 What is NF-UQ-NIDS-V3?

The **V3 datasets** (released 2025 by Luay et al. at University of Queensland) extend V2 by adding **10 temporal NetFlow features**, going from **43 features** to **53 features total**.

The V3 family includes four datasets:
- NF-UNSW-NB15-v3
- NF-ToN-IoT-v3
- NF-BoT-IoT-v3
- **NF-CSE-CIC-IDS2018-v3** ← This is the core one for your work (successor to V2)

### 2.2 V2 vs V3: What Changed

| Aspect | V2 (Your paper) | V3 (New) | Impact |
|--------|-----------------|----------|--------|
| **Features** | 43 NetFlow features | **53 features** (+10 temporal) | Richer representation |
| **Temporal features** | None | Flow timing, inter-arrival times, packet rate sequences | Captures attack patterns that evolve over time |
| **Attack classes** | 15+ | Dataset-dependent (varies) | Check specific V3 dataset |
| **Record count** | ~76M (merged) | ~2.4M (UNSW-NB15-v3 alone) | Manageable for full training |

### 2.3 The 10 New Temporal Features (V3)

From the V3 paper, the additional temporal features are:

| # | Feature | Description | Why It Matters for NIDS |
|---|---------|-------------|------------------------|
| 1 | **FW_SEQ_TIME** | Forward sequence time interval | DDoS floods show tight timing |
| 2 | **BW_SEQ_TIME** | Backward sequence time interval | Asymmetric timing can indicate exfiltration |
| 3 | **FW_STDDEV_TIME** | Std dev of forward packet timing | Scanning tools have regular timing |
| 4 | **BW_STDDEV_TIME** | Std dev of backward packet timing | Botnets show synchronized timing |
| 5 | **FW_MEAN_TIME** | Mean forward packet interval | DoS has very low mean interval |
| 6 | **BW_MEAN_TIME** | Mean backward packet interval | Benign has higher mean interval |
| 7 | **FW_MEDIAN_TIME** | Median forward packet interval | Robust to outliers |
| 8 | **BW_MEDIAN_TIME** | Median backward packet interval | Robust to outliers |
| 9 | **FW_IAT_TOTAL** | Total forward inter-arrival time | Captures flow duration structure |
| 10 | **BW_IAT_TOTAL** | Total backward inter-arrival time | Captures flow duration structure |

These features are **exactly what your ProtoNet needs** to distinguish between attacks that look similar on aggregate statistics but differ in temporal patterns (e.g., DDoS vs. DoS, scanning vs. reconnaissance).

### 2.4 NF-UNSW-NB15-v3 Distribution (Example)

| Class | Count | Percentage | Notes |
|-------|-------|------------|-------|
| Benign | 2,237,731 | 94.6% | Massively dominant |
| Exploits | 42,748 | 1.8% | Most common attack |
| Fuzzers | 33,816 | 1.4% | |
| Generic | 19,651 | 0.8% | |
| Reconnaissance | 17,074 | 0.7% | |
| DoS | 5,980 | 0.25% | |
| Shellcode | 4,659 | 0.20% | |
| Analysis | 2,381 | 0.10% | |
| Backdoor | 1,226 | 0.05% | Very rare |
| Worms | 158 | 0.007% | Extremely rare |

**Challenge:** 94.6% benign — this is even more imbalanced than V2. Your resampling strategy needs to be robust.

### 2.5 Which V3 Dataset Should You Use?

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **NF-UNSW-NB15-v3** | Good diversity, 10 classes, manageable size (2.4M) | Only one attack origin | ✅ **Primary dataset** |
| NF-CSE-CIC-IDS2018-v3 | Large, modern attacks | Very large file, may need subsampling | ⚠️ Use for cross-dataset validation |
| NF-ToN-IoT-v3 | IoT-specific attacks | Niche use case | ❌ Skip |
| NF-BoT-IoT-v3 | Botnet specific | Even more niche | ❌ Skip |

**Recommendation:** Use NF-UNSW-NB15-v3 as your **primary dataset** (well-characterized, manageable). Use NF-CSE-CIC-IDS2018-v3 for **cross-dataset generalization** tests.

### 2.6 Why V3 Makes Your Paper Stronger

1. **Novelty:** You'll be among the first to use V3 datasets (released 2025). This is a selling point.
2. **Temporal features directly help ProtoNet:** ProtoNet works best when embeddings are well-separated. Temporal features add an extra dimension of separation.
3. **The V3 paper (Luay et al., 2025) is fresh** — citing it shows you're current.
4. **You can frame your paper as** *"extending the temporal analysis of NetFlow datasets to few-shot prototypical classification"* — a natural follow-up to Luay et al.'s work.

---

## 3. Key Literature Gaps Your Paper Can Fill

| Gap | Current State | Your Opportunity |
|-----|---------------|------------------|
| **ProtoNet + temporal features** | Nobody has done this with V3 | First to combine ProtoNet with 53-feature temporal NetFlow |
| **Few-shot evaluation for NIDS** | Most "few-shot" NIDS papers don't actually evaluate on novel classes | Show 5-shot accuracy on held-out attack types |
| **Cross-dataset ProtoNet evaluation** | Every paper uses one dataset | Test NF-UNSW-NB15-v3 → NF-CSE-CIC-IDS2018-v3 transfer |
| **Production metrics for ProtoNet** | Nobody reports deployment costs for few-shot NIDS | Show KB-sized centroids, <1ms inference, Raspberry Pi viability |
| **Statistical rigor** | Most NIDS papers report single-run results | 5-fold CV, confidence intervals, McNemar's test |

---

## 4. Updated Literature Review Section (Draft)

Here's how your new Literature Review section should be structured:

```
1. Traditional ML for NIDS (2-3 paragraphs)
   - RF, XGBoost, SVM — still dominant in production
   - Key references updated to 2024-2025

2. Deep Learning for NIDS (2-3 paragraphs)
   - CNN, LSTM, hybrid models
   - The "99% accuracy" problem — most don't generalize
   - New: Temporal architectures for NIDS

3. Few-Shot Learning and Prototypical Networks (2-3 paragraphs)
   - Snell et al. (2017) — original ProtoNet
   - Niknami et al. (2024) — PTN-IDS
   - Sun et al. (2023) — ProtoCapsNet
   - YOUR OLD PAPER — acknowledge it, then describe the improvements

4. NetFlow Datasets and Temporal Features (1-2 paragraphs)
   - Sarhan et al. (2022) — V1/V2 unified feature set
   - Luay et al. (2025) — V3 temporal features
   - Why temporal features matter for NIDS

5. Research Gap and Your Contribution (1 paragraph)
   - Nobody has combined ProtoNet + temporal features + proper few-shot eval + production metrics
   - This is what your paper does
```

---

## 5. Papers to Add to Your Bibliography

Search for and add these (use Google Scholar or Semantic Scholar):

```
# ProtoNet + NIDS (must cite)
Niknami et al. 2024 - PTN-IDS
Sun et al. 2023 - Prototypical Capsule Network
Your original paper - acknowledge it

# Temporal NIDS (new area to cite)
Luay et al. 2025 - Temporal NetFlow Datasets V3
Any 2024-2025 paper on "temporal features network intrusion"

# Production NIDS (to strengthen deployment claims)
Any 2024-2025 paper on "edge deployment NIDS" or "lightweight NIDS"

# Few-shot learning advances (for context)
Any 2024-2025 NeurIPS/ICML paper on few-shot learning improvements

# Foundation models for security (cutting edge)
Any 2025 paper on "foundation model network intrusion"
```

**Target: Add 10-15 new references to your existing ~20. Total bibliography: 30-35 references.**

---

**Next:** Read [02_Production_NIDS_and_ProtoNet_Advantage.md](./02_Production_NIDS_and_ProtoNet_Advantage.md) to understand how real-world NIDS work and where ProtoNet fits.
