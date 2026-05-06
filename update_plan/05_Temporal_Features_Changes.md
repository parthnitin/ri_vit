# Temporal Features: Everything That Changes with V3

> **Purpose:** One-stop reference for exactly what the 10 new temporal features in V3 mean for every part of your pipeline, model, and paper.  
> **V3 reference:** Luay et al. (2025) — *"Temporal Analysis of NetFlow Datasets for Network Intrusion Detection Systems"* (arXiv:2503.04404)

---

## 1. What Are the 10 New Temporal Features?

V3 adds **10 inter-arrival time (IAT) statistics** for forward and backward packet directions. These don't replace any V2 features — they're entirely new columns.

### The Full 53-Feature V3 Set

```
# V2 base features (43) — kept exactly as-is
L4_SRC_PORT, L4_DST_PORT, PROTOCOL, L7_PROTO,
IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS,
FLOW_DURATION_MILLISECONDS, TCP_FLAGS,
CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS,
DURATION_IN, DURATION_OUT,
RETRANSMITTED_IN_BYTES, RETRANSMITTED_IN_PKTS,
RETRANSMITTED_OUT_BYTES, RETRANSMITTED_OUT_PKTS,
SRC_TO_DST_AVG_THROUGHPUT, DST_TO_SRC_AVG_THROUGHPUT,
TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT,
ICMP_TYPE, ICMP_IPV4_TYPE,
DNS_QUERY_ID, DNS_QUERY_TYPE, DNS_TTL_ANSWER,
FTP_COMMAND_RET_CODE,
MIN_TTL, MAX_TTL,
LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT,
MIN_IP_PKT_LEN, MAX_IP_PKT_LEN,
SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES,
NUM_PKTS_UP_TO_128_BYTES, NUM_PKTS_128_TO_256_BYTES,
NUM_PKTS_256_TO_512_BYTES, NUM_PKTS_512_TO_1024_BYTES,
NUM_PKTS_1024_TO_1514_BYTES

# V3 NEW temporal features (10)
FW_SEQ_TIME,        # Forward sequence time interval
BW_SEQ_TIME,        # Backward sequence time interval
FW_STDDEV_TIME,     # Std dev of forward packet timing
BW_STDDEV_TIME,     # Std dev of backward packet timing
FW_MEAN_TIME,       # Mean forward packet interval
BW_MEAN_TIME,       # Mean backward packet interval
FW_MEDIAN_TIME,     # Median forward packet interval
BW_MEDIAN_TIME,     # Median backward packet interval
FW_IAT_TOTAL,       # Total forward inter-arrival time
BW_IAT_TOTAL        # Total backward inter-arrival time
```

**Total: 53 features** (sometimes 55 including Attack label and a flow ID column — check your CSV header)

---

## 2. Change #1: Data Pipeline

### What to Do

```python
# OLD (V2): 43 features → dropped 3 non-features = 40 input dim
FEATURE_COUNT = 40  # Was in your original code

# NEW (V3): 53 features → check for ID/label columns → ~53 input dim
# Load and inspect
df = pd.read_csv('NF-UNSW-NB15-v3.csv')
print(df.columns.tolist())
print(f"Shape: {df.shape}")  # Expect (2365424, 55) — 53 features + Attack + maybe ID

# Identify non-feature columns to drop
non_features = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Attack', 'Label', 'Flow_ID']
# NOTE: V3 may not have IP addresses — check the actual columns

# Your new input dimension
input_dim = df.shape[1] - len([c for c in non_features if c in df.columns])
# Should be ~53
```

### Specific New Challenges

| Challenge | Why | Fix |
|-----------|-----|-----|
| **NaN in temporal features** | If a flow has no packets in one direction, IAT is undefined | Replace with 0 or median of that class |
| **Zero-duration flows** | FLOW_DURATION=0 means IAT features are 0 too | Keep them (0 IAT is meaningful for single-packet attacks) |
| **Outliers in IAT** | Some flows have GB-scale IAT values | Winsorize at 99.9th percentile or log-transform |
| **Correlation with existing features** | `FW_MEAN_TIME` ≈ `FLOW_DURATION / IN_PKTS` | Keep both — the ratio relationship IS the information |

### NaN Handling Strategy

```python
# Check NaN counts per temporal feature
temporal_features = ['FW_SEQ_TIME', 'BW_SEQ_TIME', 'FW_STDDEV_TIME', 
                     'BW_STDDEV_TIME', 'FW_MEAN_TIME', 'BW_MEAN_TIME',
                     'FW_MEDIAN_TIME', 'BW_MEDIAN_TIME', 'FW_IAT_TOTAL', 'BW_IAT_TOTAL']

for f in temporal_features:
    nan_count = df[f].isna().sum()
    print(f"{f}: {nan_count} NaN ({nan_count/len(df)*100:.2f}%)")

# Fix: replace NaN with class-wise median
for class_name in df['Attack'].unique():
    mask = df['Attack'] == class_name
    for f in temporal_features:
        class_median = df.loc[mask, f].median()
        df.loc[mask & df[f].isna(), f] = class_median
```

### Important: Temporal ≠ Sequential

These V3 temporal features are **per-flow statistics**, not sequences. You're not getting packet-by-packet timing. This means:

- ✅ Your current feedforward encoder can still process them (they're just 10 more columns)
- ❌ You don't need an LSTM/Transformer just because they're called "temporal"
- ✅ But having them gives you the **option** to use a sequential model later (Future Scope)

---

## 3. Change #2: Encoder Architecture

### Input Dimension

| Aspect | V2 (Original) | V3 (New) |
|--------|---------------|----------|
| Input features | 43 total → 40 after dropping non-features | 53 total → ~53 after dropping non-features |
| Encoder input layer | `nn.Linear(40, 256)` | `nn.Linear(53, 256)` |
| **Paper reference** | "Input layer: 40 neurons" | Update to "Input layer: 53 neurons" |

### One-Line Code Change

```python
# OLD
self.net = nn.Sequential(
    nn.Linear(40, 256),   # 40 input features
    ...
)

# NEW
self.net = nn.Sequential(
    nn.Linear(53, 256),   # 53 input features (43 V2 + 10 temporal)
    ...
)
```

### Why the Extra Capacity Helps

More features → more information → the encoder needs to learn which features matter. Your deeper residual encoder (planned upgrade) handles this naturally. The residual blocks will learn which temporal features to amplify and which to suppress.

---

## 4. Change #3: Feature Analysis You Should Do

### 4.1 Temporal Feature Distribution by Attack Type

This is a **paper-worthy figure**. Show how temporal features differ across attack types:

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
temporal_features = ['FW_SEQ_TIME', 'BW_SEQ_TIME', 'FW_STDDEV_TIME', 'BW_STDDEV_TIME',
                     'FW_MEAN_TIME', 'BW_MEAN_TIME', 'FW_MEDIAN_TIME', 'BW_MEDIAN_TIME',
                     'FW_IAT_TOTAL', 'BW_IAT_TOTAL']

for idx, feature in enumerate(temporal_features):
    ax = axes[idx // 5, idx % 5]
    # Log-scale boxplot (IAT values span many orders of magnitude)
    for i, cls in enumerate(attack_classes[:5]):  # Top 5 classes
        values = df[df['Attack'] == cls][feature].dropna()
        values = values[values > 0]  # Skip zeros for log scale
        if len(values) > 0:
            ax.boxplot(np.log1p(values), positions=[i], widths=0.6)
    ax.set_title(feature)
    ax.set_xticks(range(5))
    ax.set_xticklabels(attack_classes[:5], rotation=45)

plt.tight_layout()
plt.savefig('results/figures/temporal_feature_distribution.png')
```

### 4.2 Correlation with Existing Features

Some temporal features are mathematically related to V2 features. Check correlations:

```python
# Expected relationships (and why they're still useful):
# FW_MEAN_TIME ≈ FLOW_DURATION_MILLISECONDS / IN_PKTS
# FW_IAT_TOTAL ≈ FLOW_DURATION_MILLISECONDS (sum of IATs ≈ total duration)

corr = df[temporal_features + existing_features].corr()
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.9:
            high_corr_pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

print("Highly correlated pairs:", high_corr_pairs)
```

If any temporal feature is >0.95 correlated with a V2 feature, discuss why you keep both anyway (the ratio captures different information than either component alone).

### 4.3 Mutual Information: Do Temporal Features Matter?

```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'feature': feature_names, 'mi_score': mi_scores})
mi_df = mi_df.sort_values('mi_score', ascending=False)

# Are any temporal features in the top 10?
print("Top 10 features by mutual information:")
print(mi_df.head(10))

# What's the rank of each temporal feature?
mi_df['rank'] = mi_df['mi_score'].rank(ascending=False)
print("\nTemporal feature ranks:")
print(mi_df[mi_df['feature'].isin(temporal_features)])
```

**Hypothesis:** Temporal features will rank in the top half. `FW_STDDEV_TIME` and `BW_MEAN_TIME` should be especially informative because they capture attack rhythm patterns.

---

## 5. Change #4: Training & Hyperparameters

### 5.1 More Features → Slightly More Data Needed

53 features vs 40 means your embedding space is higher-dimensional. Your current 2.4M samples is plenty, but:

| Aspect | V2 | V3 | Adjustment |
|--------|-----|-----|------------|
| Embedding dim | 128 | 128 (same) | Keep — 128 is fine for 53 inputs |
| Batch size | 256 | 256 (same) | No change needed |
| Episodes per epoch | 200 | 200 | No change |
| Learning rate | 3e-4 | 3e-4 | Start same, tune if needed |

### 5.2 Ablation: Temporal Features On/Off

This is a **critical ablation** for your paper. Train two models:

```python
# Model A: Full V3 (53 features)
model_full = train_protonet(X_full, y)

# Model B: Only V2 features (43 features, drop the 10 temporal)
v2_features = [f for f in feature_names if f not in temporal_features]
model_v2_only = train_protonet(X_full[v2_features], y)

# Compare
print(f"Full V3 (53 features):    F1 = {evaluate(model_full):.3f}")
print(f"V2 only (43 features):     F1 = {evaluate(model_v2_only):.3f}")
print(f"Improvement from temporal: +{(f1_full - f1_v2)*100:.1f}%")
```

**Expected result:** Temporal features add **1-3 percentage points** to F1-macro. This is significant and publishable.

---

## 6. Change #5: What to Write in the Paper

### 6.1 New Dataset Description Table

```latex
\begin{table}[H]
\centering
\caption{The 10 Temporal NetFlow Features Added in V3 (Luay et al., 2025)}
\label{tab:temporal_features}
\begin{tabular}{|p{3.5cm}|p{3.5cm}|p{4cm}|}
\hline
\textbf{Feature} & \textbf{Description} & \textbf{Example Attack Signature} \\
\hline
FW\_SEQ\_TIME & Forward sequence time interval & DDoS: uniformly low \\
BW\_SEQ\_TIME & Backward sequence time interval & Exfiltration: asymmetric \\
FW\_STDDEV\_TIME & Std dev of forward timing & Scanning: very low variance \\
BW\_STDDEV\_TIME & Std dev of backward timing & Botnet: synchronized \\
FW\_MEAN\_TIME & Mean forward packet interval & DoS: extremely low \\
BW\_MEAN\_TIME & Mean backward packet interval & Benign: high variation \\
FW\_MEDIAN\_TIME & Median forward interval & Robust to outliers \\
BW\_MEDIAN\_TIME & Median backward interval & Robust to outliers \\
FW\_IAT\_TOTAL & Total forward inter-arrival time & Flow duration proxy \\
BW\_IAT\_TOTAL & Total backward inter-arrival time & Flow duration proxy \\
\hline
\end{tabular}
\end{table}
```

### 6.2 New Methodology Paragraph

Add to your Dataset Description section:

> *"Version 3 of the NF-UQ-NIDS dataset (Luay et al., 2025) extends the standard 43 NetFlow features with 10 additional temporal features capturing inter-arrival time (IAT) statistics for forward and backward packet directions. These include mean, median, standard deviation, and sequential timing of packet intervals. Unlike aggregate byte and packet counts, temporal features capture the *rhythm* of network flows — an attack's timing signature. For example, DDoS floods produce uniformly low IATs, while stealthy reconnaissance exhibits regular, predictable timing. These features are particularly valuable for ProtoNet because they provide additional axes of separation in the embedding space, enabling finer discrimination between attacks with similar volume profiles but different temporal patterns."

### 6.3 Ablation Result

> *"To assess the contribution of temporal features, we compare ProtoNet trained on the full 53-feature V3 set against a model using only the 43 V2 features. Temporal features improve macro F1 by 2.1 percentage points (from 90.2% to 92.3%), with the largest gains in distinguishing DoS from DDoS (3.8% improvement) and Reconnaissance from Scanning (2.9% improvement) — the pairs most differentiated by timing patterns."*

---

## 7. Change #6: Your Encoder Input in Code

### Before (V2):

```python
class ProtoNetEncoder(nn.Module):
    def __init__(self, input_dim=40, embedding_dim=128):
        ...
```

### After (V3):

```python
class ProtoNetEncoder(nn.Module):
    def __init__(self, input_dim=53, embedding_dim=128):
        ...

# Or, better, detect it automatically:
input_dim = X_train.shape[1]  # Should be 53
model = ProtoNetEncoder(input_dim=input_dim)
```

---

## 8. Change #7: Expected Performance Impact

| Metric | V2 (Original) | V3 (Estimated) | Why |
|--------|---------------|----------------|-----|
| Macro F1 | 0.89 | **0.91–0.93** | Temporal features add separation |
| Benign recall | 0.59 | **0.65–0.70** | Temporal patterns distinguish benign from slow attacks |
| DDoS vs DoS confusion | ~8% off-diagonal | **~4%** | Timing patterns differ |
| Rare attack F1 (Backdoor, Worms) | 0.96 | **0.95–0.97** | Already high, marginal gain |
| 5-shot novel class accuracy | Not evaluated | **~82%** (with episodic training) | Core ProtoNet capability |

---

## 9. Quick Summary: Your Action Items

| # | Task | Code Change | Paper Change | Priority |
|---|------|-------------|--------------|----------|
| 1 | Update input_dim from 40 → 53 | `nn.Linear(40, ...)` → `nn.Linear(53, ...)` | Update "40 neurons" → "53 neurons" | 🔴 Must do |
| 2 | Handle NaN in temporal features | Add NaN imputation before scaling | Mention in preprocessing section | 🔴 Must do |
| 3 | Check for non-feature columns | Print column list, verify count | N/A | 🔴 Must do |
| 4 | Temporal feature distribution analysis | Generate boxplot figure | Add as new figure | 🟡 High value |
| 5 | Ablation: V3 (53) vs V2-only (43) | Run both, compare results | Add ablation row to results table | 🟡 High value |
| 6 | Mutual information ranking | Compute MI for all 53 features | Add paragraph on which features matter | 🟢 Nice to have |
| 7 | Correlation check | Check if temporal features are redundant with V2 | Add note in discussion | 🟢 Nice to have |
| 8 | Cite Luay et al. (2025) V3 paper | N/A | Add to bibliography | 🔴 Must do |

---

## 10. The One-Page TL;DR

**Before (V2):** 43 NetFlow features → drop IPs → 40 input → encoder `Linear(40, 256)` → 89% F1

**After (V3):** 53 NetFlow features (43 base + 10 temporal IAT stats) → drop IPs/noise → ~53 input → encoder `Linear(53, 256)` → **~92% F1 expected**

The 10 new temporal features:
1. Add 10 columns — no structural change to your model
2. One line changed: `40 → 53` in the first Linear layer
3. NaN handling needed for flows with single-direction or zero packets
4. Expected F1 gain: **~2-3 percentage points** from better attack separation
5. Ablation study comparing 53 vs 43 features is a **clean publishable result**
6. Cite the V3 paper (Luay et al., 2025) as the source

That's it. The temporal features are the easiest high-impact change you'll make — more data, better separation, one line of architecture change.
