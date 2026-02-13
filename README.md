# Contextual Bandit News Recommendation System

**Course:** Reinforcement Learning Fundamentals  
**Student:** Vaibhav Dabas  
**Roll Number:** U20230099  

This project implements a **Contextual Multi-Armed Bandit (CMAB) based News Recommendation System** where:

- User category acts as the context
- News category acts as the arm
- Rewards simulate user engagement
- Machine learning predicts user context
- Reinforcement learning selects optimal recommendations

The objective is to maximize user engagement by recommending the best news category conditioned on user context.

---

# Overall Approach & Design Decisions

## 1. Data Preprocessing

Steps performed:

- Missing age values replaced using median imputation.
- Frequency encoding applied to categorical features.
- Subscriber converted to numeric form.
- StandardScaler used for normalization.
- Non-predictive columns removed.

### Design Rationale

- Improves model convergence.
- Prevents bias due to missing data.
- Reduces dimensionality.
- Ensures stable classification performance.

---

## 2. User Context Classification

A **Gradient Boosting Classifier** was used to predict user category:

- user_1
- user_2
- user_3

### Why Gradient Boosting?

- Strong performance on tabular datasets.
- Handles nonlinear feature interactions.
- Robust against noise.
- Provides stable predictions.

### Performance

- Validation Accuracy ≈ **91.5%**
- Training Accuracy ≈ **100%**

This predicted context serves as input for the bandit algorithms.

---

## 3. Contextual Bandit Algorithms

Three exploration strategies were implemented.

---

### Epsilon-Greedy

Mechanism:

- With probability ε → explore random arm.
- Otherwise → exploit best estimated arm.

Hyperparameters tested:

- ε = 0.01
- ε = 0.1
- ε = 0.3

Key tradeoff:

- Low ε → better exploitation.
- High ε → excessive exploration.

---

### Upper Confidence Bound (UCB)

Formula:

`UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))`

Where:

- Q(a) = Estimated reward
- N(a) = Number of pulls
- c = Exploration parameter

Hyperparameters tested:

- c = 0.5
- c = 1.0
- c = 2.0

Advantages:

- Confidence-based exploration.
- Fast convergence.
- Strong theoretical guarantees.

---

### SoftMax Exploration

Probability:

`P(a) ∝ exp(Q(a)/τ)`

Where:

- τ controls randomness.

Tested:

- τ = 0.1
- τ = 1.0
- τ = 5.0

Behaviour:

- Low τ → greedy selection.
- High τ → near-random exploration.

---

## 4. Recommendation Pipeline

End-to-end workflow:

1. Predict user context using classifier.
2. Select optimal news category using learned bandit policy.
3. Randomly sample an article from that category.
4. Output recommendation.

Best policy observed:

- user_1 → Education
- user_2 → Tech
- user_3 → Entertainment

---

# Key Results

## Classification Results

- Validation Accuracy ≈ **91.5%**
- Reliable context detection achieved.

---

## Mean Rewards (T = 10,000 Steps, Best Hyperparameters)

| Context | Epsilon-Greedy (ε=0.01) | UCB (c≈1.0) | SoftMax (τ=0.1) |
|--------|--------------------------|-------------|----------------|
| user_1 | ~4.4 | ~4.45 | ~4.45 |
| user_2 | ~10.1 | ~10.2 | ~10.2 |
| user_3 | ~7.2 | ~7.3 | ~7.29 |

**Best Overall Algorithm:** UCB

---

## Observations

### Algorithm Performance

- UCB consistently achieved highest rewards.
- SoftMax with low τ performed similarly to UCB.
- Epsilon-Greedy slightly lower due to continuous exploration.

---

### Hyperparameter Effects

#### Epsilon-Greedy

- Smaller ε improves exploitation.
- Larger ε decreases reward due to random pulls.

#### UCB

- Robust to parameter changes.
- Converges quickly across contexts.

#### SoftMax

- Highly sensitive to temperature τ.
- Low τ performs best.

---

### Context Impact

Reward variation by context:

- user_2 → highest reward (~10)
- user_3 → moderate (~7)
- user_1 → lowest (~4)

This confirms the importance of contextual modeling.

---

### Exploration vs Exploitation

- Excess exploration reduces performance.
- Balanced exploration improves convergence.

---

### Recommendation Distribution

Approximate recommendations:

- Tech ≈ 696
- Education ≈ 663
- Entertainment ≈ 641

Shows balanced contextual recommendations.

---

# Reproducing the Experiments

## Step 1 — Install Dependencies

Run:

`pip install rlcmab-sampler scikit-learn pandas matplotlib seaborn`

---

## Step 2 — Dataset Setup

Create directory:

`data/`

Place:

- `news_articles.csv`
- `train_users.csv`
- `test_users.csv`

inside the folder.

---

## Step 3 — Run Project

Run script:

`python main.py`

OR execute notebook:

`lab3_results_U20230099.ipynb`

---

## Step 4 — Outputs Generated

You will obtain:

- User classification predictions
- Bandit simulation results
- Average reward plots
- Hyperparameter comparison plots
- Final recommendations dataset

---

# Final Conclusion

This project demonstrates that:

- Contextual bandits significantly improve recommendation quality.
- Accurate context prediction enhances personalization.
- UCB provides the best exploration-exploitation balance.
- Controlled exploration leads to optimal reward convergence.

Contextual reinforcement learning is highly effective for real-world recommendation systems such as news platforms, e-commerce, and personalized content delivery.

---
