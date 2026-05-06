# Implementation Plan

> **Purpose:** Concrete week-by-week code and writing roadmap. What to build, what order, what "done" looks like.  
> **Read after:** All other documents

---

## Overview

```
Phase 0: Setup                    Week 0     (1-2 days)
Phase 1: Data Pipeline            Week 1     (5-7 days)
Phase 2: Model Implementation     Week 2-3   (10-14 days)
Phase 3: Experiments              Week 4-5   (10-14 days)
Phase 4: Paper Writing            Week 6-7   (10-14 days)
Phase 5: Mentor Contact           Day 1 of Week 8
```

**Total: ~6-7 weeks of focused weekend/evening work.**

---

## Phase 0: Setup (Week 0, 1-2 days)

### Goal
Get your development environment ready so you don't waste time later.

### Tasks

- [ ] **Install Python 3.10+ and PyTorch 2.x**
- [ ] **Set up project structure:**

```
nids_protonet_v2/
├── data/
│   ├── raw/           # Downloaded V3 datasets
│   └── processed/     # Preprocessed NumPy/PyTorch tensors
├── src/
│   ├── data.py        # Data loading, preprocessing, episode sampling
│   ├── models.py      # Encoder, ProtoNet classes
│   ├── train.py       # Training loop with episodic sampling
│   ├── evaluate.py    # All evaluation metrics
│   ├── baselines.py   # RF, XGBoost, MLP baselines
│   └── deploy.py      # Production benchmarking
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_training.ipynb
│   └── 03_results.ipynb
├── paper/
│   └── paper_v2.tex   # New paper
└── README.md
```

- [ ] **Download V3 datasets:**
  - NF-UNSW-NB15-v3 (primary)
  - NF-CSE-CIC-IDS2018-v3 (cross-dataset)
  - URL: https://staff.itee.uq.edu.au/marius/NF_UQ_NIDS_dataset/

- [ ] **Install key libraries:**
  ```bash
  pip install torch numpy pandas scikit-learn xgboost imbalanced-learn 
  pip install matplotlib seaborn optuna  # hyperparameter search
  pip install onnx onnxruntime  # for production export
  ```

- [ ] **Run a quick sanity check:**
  Load the V3 CSV, check it has 53 features, plot class distribution.

### Deliverable
A working Python environment with data loaded. Takes 1-2 evenings.

---

## Phase 1: Data Pipeline (Week 1, 5-7 days)

### Goal
Build a clean, correct data pipeline with proper preprocessing and episode sampling.

### Tasks

#### Day 1-2: Data Loading and Exploration

- [ ] Load NF-UNSW-NB15-v3 CSV (2.4M rows × 53 features)
- [ ] Check for: missing values, NaN, infinite values
- [ ] Feature analysis:
  - Which features are numerical vs categorical?
  - Which are temporally correlated?
  - Identify constant/near-constant features to drop
- [ ] Class distribution analysis (you already know it's imbalanced — quantify it)

#### Day 3-4: Preprocessing

- [ ] Drop irrelevant features (IP addresses, timestamps — keep temporal stats)
- [ ] Handle missing values (median imputation for numerical features)
- [ ] **Stratified shuffle** the dataset (critical fix from methodology review)
- [ ] Split: 60/20/20 stratified train/val/test
- [ ] Min-Max scaling (fit on train only, transform all)
- [ ] Encode labels as integers
- [ ] Save processed data as PyTorch tensors for fast loading

#### Day 5-6: Episode Sampler

Build the episode sampler class:

```python
class EpisodeSampler:
    """Samples N-way K-shot episodes for ProtoNet training."""

    def __init__(self, labels, n_way=10, k_shot=5, k_query=5):
        # Group indices by class
        # Filter out classes with < k_shot + k_query samples
        pass

    def sample_episode(self):
        # Returns: support_x, support_y, query_x, query_y
        pass
```

**Test it thoroughly:**
- Run 100 episodes, verify shapes are correct
- Verify labels match between support and query for same class
- Verify no data leakage (support/query are disjoint within episode)

#### Day 7: Data Pipeline Integration

- [ ] Create `data.py` with:
  - `load_and_preprocess(path)` → returns X_train, y_train, X_val, y_val, X_test, y_test
  - `get_episode_sampler(labels, n_way, k_shot, k_query)` → returns EpisodeSampler
  - `get_dataloader(X, y, batch_size)` → for standard evaluation

### Deliverable
A script that loads V3 data, preprocesses correctly, and produces episode batches for training.

```
src/data.py
    load_and_preprocess() → (X_train, y_train, X_val, y_val, X_test, y_test)
    EpisodeSampler.sample_episode() → (support_x, support_y, query_x, query_y)
```

---

## Phase 2: Model Implementation (Weeks 2-3, 10-14 days)

### Goal
Build the encoder and training loop. Get a working model that converges.

### Tasks

#### Week 2, Day 1-2: Encoder Model

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class ProtoNetEncoder(nn.Module):
    def __init__(self, input_dim=53, embedding_dim=128):
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)
```

#### Week 2, Day 3-5: Training Loop

```python
def train_epoch(model, episode_sampler, optimizer, device, num_episodes=200):
    model.train()
    total_loss = 0

    for _ in range(num_episodes):
        support_idx, support_y, query_idx, query_y = episode_sampler.sample_episode()

        # Get embeddings
        support_emb = model(X[support_idx].to(device))  # (N*K, D)
        query_emb = model(X[query_idx].to(device))       # (N*Q, D)

        # Compute centroids from support set
        centroids = []
        for c in range(episode_sampler.n_way):
            mask = torch.tensor(support_y) == c
            centroids.append(support_emb[mask].mean(0))
        centroids = torch.stack(centroids)  # (N, D)

        # Compute distances from queries to centroids
        distances = torch.cdist(query_emb, centroids, p=2)  # (N*Q, N)
        logits = -distances  # Negative distance = similarity
        loss = F.cross_entropy(logits, torch.tensor(query_y).to(device))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_episodes
```

#### Week 2, Day 6-7: Validation Loop

```python
def evaluate(model, X_val, y_val, centroids, device):
    model.eval()
    with torch.no_grad():
        embeddings = model(X_val.to(device))
        distances = torch.cdist(embeddings, centroids.to(device), p=2)
        predictions = distances.argmin(dim=1).cpu().numpy()

    accuracy = accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions, average='macro')
    report = classification_report(y_val, predictions, output_dict=True)
    return accuracy, f1, report
```

#### Week 3, Day 1-3: Full Training Pipeline

- [ ] Add learning rate scheduling (CosineAnnealingLR or ReduceLROnPlateau)
- [ ] Add model checkpointing (save best based on validation loss)
- [ ] Add early stopping (patience = 10 epochs)
- [ ] TensorBoard logging (or just print metrics)
- [ ] Run first full training — does it converge?

#### Week 3, Day 4-7: Hyperparameter Search

Use Optuna for a basic search:

```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    n_way = trial.suggest_categorical('n_way', [5, 10, 15])
    k_shot = trial.suggest_categorical('k_shot', [1, 3, 5, 10])

    # Train and return validation F1
    ...
```

Run 30-50 trials. Report the best hyperparameters.

### Deliverable

A working training pipeline that:
- Loads V3 data
- Trains with episodic sampling
- Achieves >90% F1-macro on validation
- Saves model checkpoints and centroids

---

## Phase 3: Experiments (Weeks 4-5, 10-14 days)

### Goal
Run all experiments needed for the paper: cross-validation, baselines, few-shot, ablation, production metrics.

### Tasks

#### Week 4, Day 1-3: Cross-Validation

- [ ] Implement 5-fold stratified cross-validation
- [ ] Train on each fold, evaluate on held-out fold
- [ ] Report: mean ± std for all metrics across folds

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    # Train on train_idx, evaluate on test_idx
    accuracy, f1, report = run_experiment(X[train_idx], y[train_idx],
                                          X[test_idx], y[test_idx])
    fold_results.append({'accuracy': accuracy, 'f1': f1})

print(f"Accuracy: {np.mean([r['accuracy'] for r in fold_results]):.3f} "
      f"± {np.std([r['accuracy'] for r in fold_results]):.3f}")
```

#### Week 4, Day 4-5: Baselines

Run these on **your exact train/test split**:

- [ ] **Random Forest** — `RandomForestClassifier(n_estimators=200, random_state=42)`
- [ ] **XGBoost** — `XGBClassifier(n_estimators=200, random_state=42)`
- [ ] **MLP** — `MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200)`
- [ ] **Logistic Regression** — simple linear baseline

For each: collect accuracy, F1-macro, F1-per-class, training time, model size.

#### Week 4, Day 6-7: Few-Shot Evaluation

- [ ] Hold out 3-5 attack classes from training entirely
- [ ] For each held-out class, sample K support examples (1, 3, 5, 10)
- [ ] Test on Q query examples (e.g., 50 per class)
- [ ] Report accuracy vs. number of shots

```
Held-out classes: Worms, Shellcode, Backdoor
Results:
  - 1-shot: 68.5% accuracy on novel classes
  - 3-shot: 76.2%
  - 5-shot: 82.1%
  - 10-shot: 85.7%
```

#### Week 5, Day 1-2: Ablation Studies

Run these variants:

- [ ] No episodic training (standard batch training with centroid loss)
- [ ] No residual blocks (2-layer MLP from original paper)
- [ ] No temporal features (use only 43 V2 features from V3 dataset)
- [ ] No SMOTE/focal loss (raw class distribution)
- [ ] Different distance metrics (cosine, L1, L2)

Collect accuracy and F1 for each.

#### Week 5, Day 3-4: Embedding Space Analysis

- [ ] Silhouette score for each class
- [ ] Davies-Bouldin index
- [ ] Inter/Intra class distance ratio
- [ ] t-SNE visualization of embeddings (colored by class)
- [ ] Centroid stability across 5 training runs (different seeds)

#### Week 5, Day 5-6: Production Metrics

- [ ] Inference latency (P50, P95, P99) — single sample, on CPU
- [ ] Batch throughput (batch sizes 1, 16, 64, 256)
- [ ] Model size (encoder + centroids)
- [ ] **Bonus:** Run on Raspberry Pi if you have one
- [ ] **Bonus:** Export to ONNX and measure the speedup

#### Week 5, Day 7: Statistical Significance

- [ ] Train ProtoNet and XGBoost with 5 different seeds
- [ ] Run paired bootstrap test: is ProtoNet's minority F1 significantly better?
- [ ] Run McNemar's test: do ProtoNet and XGBoost make different errors?

### Deliverable
All results tables populated. Ready to write the paper.

---

## Phase 4: Paper Writing (Weeks 6-7, 10-14 days)

### Goal
Write the complete LaTeX paper. Generate all figures.

### Tasks

#### Week 6, Day 1-2: Structure and Draft Figures

- [ ] Set up the LaTeX document structure
- [ ] Generate all figures from experiment results
  - Confusion matrix (Normalized)
  - 3D centroid visualization (t-SNE or PCA + centroids)
  - Ablation bar chart
  - Few-shot accuracy curve (accuracy vs. number of shots)
  - Latency comparison bar chart

#### Week 6, Day 3-5: Write Sections

Using the [Full Paper Rewrite Strategy](./03_Full_Paper_Rewrite_Strategy.md):

- [ ] Abstract
- [ ] Introduction
- [ ] Literature Review (with proper citations to new papers)
- [ ] Problem Statement

#### Week 6, Day 6-7: Write Methodology and Results

- [ ] Methodology (with V3 dataset, new encoder, episodic training)
- [ ] Results (all tables with ± std)
- [ ] Few-shot and production results subsections

#### Week 7, Day 1-2: Write Discussion and Conclusion

- [ ] Discussion (fair comparison, ablation, limitations)
- [ ] Future Scope
- [ ] Conclusion

#### Week 7, Day 3-4: Reference Management

- [ ] Finalize bibliography (30-35 references)
- [ ] Check all citations are correctly formatted

#### Week 7, Day 5-6: Polish

- [ ] Proofread everything
- [ ] Check all claims match actual results
- [ ] Verify all figures are clear and properly labeled
- [ ] Check page limit (~20 pages)

#### Week 7, Day 7: Final Review

- [ ] Read the entire paper from start to finish
- [ ] Have a friend/colleague read it
- [ ] Fix all typos and unclear phrasing

### Deliverable
Complete LaTeX paper, ready for submission and/or mentor review.

---

## Phase 5: Mentor Contact (Week 8, Day 1)

### Goal
Send the completed paper to your mentor professionally.

### Tasks

- [ ] Write the email (use template from [00_READ_THIS_FIRST.md](./00_READ_THIS_FIRST.md))
- [ ] Attach: paper PDF + any supplementary materials
- [ ] Send and wait for response

### Possible Follow-ups

| Response | Action |
|----------|--------|
| "Yes, let's co-author" | Ask about preferred journal, set revision timeline |
| "Busy, maybe later" | Thank him, ask to proceed independently. Offer co-authorship or acknowledgment |
| "No" or no response | Proceed independently. Acknowledge his original supervision |
| "Need major changes" | Evaluate. Legitimate feedback → incorporate. Nitpicking → decide if worth it |

---

## Quick Reference: What "Done" Looks Like

| Component | Done When |
|-----------|-----------|
| **Data pipeline** | `python src/data.py` runs without errors, produces correct episode batches |
| **Training** | Model converges to >90% F1-macro on validation |
| **5-fold CV** | Results table with mean ± std across folds |
| **Baselines** | RF, XGBoost, MLP results on same split |
| **Few-shot evaluation** | Held-out class accuracy vs. number of shots |
| **Ablation studies** | Table showing impact of each component |
| **Production metrics** | Latency, throughput, model size measured |
| **Statistical tests** | Bootstrap CI and McNemar's p-value |
| **Paper draft** | Complete LaTeX with all sections filled |
| **Figures** | Confusion matrix, centroid viz, ablation chart, few-shot curve, latency chart |
| **Final review** | Paper read end-to-end, no typos, claims match results |

---

## Code Repository Structure (Final)

```
nids_protonet_v2/
├── README.md                        # Overview, how to run
├── requirements.txt                 # Python dependencies
├── Makefile                         # Common commands (make train, make evaluate)
│
├── data/
│   ├── raw/                         # Downloaded CSV files (not in git)
│   │   ├── NF-UNSW-NB15-v3.csv
│   │   └── NF-CSE-CIC-IDS2018-v3.csv
│   └── processed/                   # Preprocessed tensors (not in git)
│       ├── train_data.pt
│       ├── train_labels.pt
│       ├── val_data.pt
│       ├── val_labels.pt
│       ├── test_data.pt
│       └── test_labels.pt
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # All hyperparameters in one place
│   ├── data.py                      # Data loading, preprocessing, episode sampler
│   ├── models.py                    # Encoder, ProtoNet, loss functions
│   ├── train.py                     # Training and validation loops
│   ├── evaluate.py                  # All evaluation metrics
│   ├── baselines.py                 # RF, XGBoost, MLP implementations
│   └── deploy.py                    # Production benchmarking
│
├── experiments/
│   ├── 01_train_protonet.py         # Run training
│   ├── 02_cross_validation.py       # Run 5-fold CV
│   ├── 03_baselines.py              # Run baseline comparisons
│   ├── 04_few_shot_eval.py          # Run few-shot evaluation
│   ├── 05_ablation.py               # Run ablation studies
│   └── 06_production_benchmark.py   # Run deployment metrics
│
├── results/
│   ├── trained_model.pt             # Best model checkpoint
│   ├── centroids.pt                 # Learned centroids
│   ├── results.json                 # All experiment results
│   └── figures/                     # Generated figures
│       ├── confusion_matrix.png
│       ├── centroid_viz.png
│       ├── ablation.png
│       ├── few_shot_curve.png
│       └── latency_comparison.png
│
└── paper/
    └── paper_v2.tex                 # New paper LaTeX source
```

---

## Final Word

You've got a clear plan. The hard part isn't the ML — it's the **discipline to execute consistently** over 6-7 weeks.

**My advice:**
1. Commit to **30 minutes every day** rather than 4 hours on weekends. Consistency beats intensity.
2. Start each week by reading that week's section of the plan. End each week by checking off the tasks.
3. If something takes twice as long as expected, **that's normal**. Just adjust the timeline.
4. The mentor email is the last step, not the first. Do the work, then share it.

Good luck. The core ProtoNet idea is solid — you just need to execute on the upgrades. 🚀
