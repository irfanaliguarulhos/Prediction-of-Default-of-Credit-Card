# 🏦 Banking Identity-Ring Fraud Detection — Complete Interview Q&A
## Person Embeddings + Classification (PySpark)

> **Prepared for:** Technical, Stakeholder, Executive, and CEO-level interviews  
> **Project:** End-to-End Fraud Detection Pipeline with PySpark, Embeddings & Classification  
> **Business Question:** How can we minimize direct financial losses from coordinated identity fraud while protecting legitimate customer experience?

---

## 📋 TABLE OF CONTENTS

| Section | Audience | Topics Covered |
|---------|----------|---------------|
| 1. Architecture & Stakeholder Alignment | All Levels | System design, trade-offs, KPI alignment |
| 2. Data Ingestion & Quality | Technical | PySpark, messy data, schema enforcement |
| 3. EDA & Visualization | Technical | Fraud patterns, graph analysis, sampling |
| 4. Feature Engineering & Embeddings | Technical | Person embeddings, graph features, Node2Vec |
| 5. Model Selection & Training | Technical | LightGBM, GNN, Neural Networks, ensembles |
| 6. Evaluation & Business Metrics | All Levels | Cost-aware metrics, threshold optimization |
| 7. Hyperparameter Tuning | Technical | Bayesian optimization, distributed tuning |
| 8. Explainability & Fairness | All Levels | SHAP, bias mitigation, human-in-the-loop |
| 9. Deployment & Serving | Technical | REST/gRPC, feature stores, canary rollout |
| 10. Monitoring & Drift Detection | All Levels | PSI, performance drift, alerting |
| 11. Testing & Validation | Technical | Unit tests, A/B testing, statistical tests |
| 12. CEO & Executive Questions | CEO/Executive | ROI, risk, competitive advantage |
| 13. Stakeholder & Product Questions | Product/Growth | Friction vs fraud, UX trade-offs |
| 14. Risk & Compliance Questions | Risk/Ops | Regulatory, audit, operational metrics |
| 15. Scenario-Based Questions | All Levels | Real-world problem solving |
| 16. Quick Reference: Model Comparison | All Levels | Decision matrix for model selection |

---

# 1. ARCHITECTURE & STAKEHOLDER ALIGNMENT

## 🔹 For Technical Interviewers

### Q1.1: Walk me through the end-to-end architecture of this fraud detection system. What are the key components?

**A:** The architecture follows a Lambda pattern with batch and real-time paths:

```
DATA SOURCES
  Raw Logs | Transactions | KYC | Device Signals | Blacklists
       ↓
INGESTION LAYER (PySpark)
  Schema Enforcement | Partitioning | Deduplication | Bronze
       ↓
FEATURE STORE (Parquet/Delta Lake)
  Time-windowed Aggregates | Graph Features | Embeddings | Silver
       ↓
MODEL TRAINING PIPELINE
  Spark MLlib | LightGBM/XGBoost | PyTorch/TensorFlow | GraphSAGE
       ↓
SERVING LAYER
  Batch Scoring (nightly) | Real-time REST/gRPC | Model Server
       ↓
MONITORING & FEEDBACK
  Drift Detection | SHAP Explainability | Alerting | Dashboard
```

**Key Design Decisions:**
- **PySpark for scale:** Handles TB-scale data with fault-tolerant distributed processing
- **Feature store:** Decouples feature engineering from training/serving to prevent training-serving skew
- **Dual serving:** Batch for backfills and reporting; real-time for transaction decisioning
- **Delta Lake:** ACID transactions and time travel for reproducibility

---

### Q1.2: Why did you choose PySpark over pandas or Dask for this project?

**A:** 

| Dimension | PySpark | pandas | Dask |
|-----------|---------|--------|------|
| **Scale** | PB-level across clusters | Single machine (GBs) | Moderate distributed |
| **Fault Tolerance** | Built-in lineage & recomputation | None | Limited |
| **Integration** | Native Parquet/Delta, MLlib | Manual I/O | Partial |
| **Graph Processing** | GraphFrames/GraphX | NetworkX (single node) | Limited |
| **Streaming** | Structured Streaming | None | Limited |
| **When to use** | Production pipelines | EDA, prototyping | Medium-scale ETL |

**Trade-off:** PySpark has higher latency for small data but is essential for production-scale fraud detection where we process millions of transactions daily with complex graph joins.

---

### Q1.3: How do you handle the "cold start" problem for new accounts with no transaction history?

**A:** Multi-layered approach:
1. **KYC signals:** Name/address/phone risk scores from external databases
2. **Device fingerprinting:** New device = higher risk; known device = lower risk
3. **Graph linkage:** If new account shares device/IP with known fraud ring → elevated risk
4. **Progressive profiling:** Start with strict limits, relax as behavior establishes
5. **Ensemble with unsupervised model:** Isolation Forest flags anomalies even without labels
6. **Default risk tier:** Assign to "medium" risk with step-up verification

---

### Q1.4: What partitioning strategy do you use for the transaction data, and why?

**A:** 
```python
# Partition by event_date for time-travel and efficient pruning
clean.write.mode("append") \
    .partitionBy("event_date") \
    .parquet("/data/bronze/transactions")
```

**Rationale:**
- **Time-based queries** ("fraud in last 7 days") prune 99% of partitions
- **Backfills** only touch relevant date ranges
- **Retention policies** drop old partitions atomically
- **Bucketing** on account_id within partitions for efficient joins

**Secondary bucketing:** account_id bucketed into 256 buckets for join optimization with account/KYC data.

---

## 🔹 For Stakeholder/Executive Interviewers

### Q1.5: How does this system align with our three stakeholder groups (Risk, Product, Engineering)?

**A:** 

| Stakeholder | Primary KPI | How System Delivers |
|-------------|-------------|---------------------|
| **Risk & Ops** | Gross Fraud Loss | Real-time blocking + graph detection catches rings before they scale |
| **Product & Growth** | Customer Friction Rate | Two-stage scoring (high-recall → precision) minimizes false positives |
| **Data Science/Eng** | Latency & Scalability | PySpark distributed + ONNX serving achieves <50ms p99 latency |

**The tension:** Risk wants to block everything; Product wants to approve everything. The system resolves this through **cost-aware threshold optimization** — we mathematically minimize total expected cost (fraud loss + friction cost), not just accuracy.

---

### Q1.6: What is the expected business impact in the first 6 months?

**A:** 

**Phase 1 (Months 1-2):** Baseline establishment
- Deploy rule-based + logistic regression baseline
- Expected: 15-20% fraud loss reduction
- Cost: ~$200K infrastructure + 2 FTEs

**Phase 2 (Months 3-4):** ML model rollout
- LightGBM with graph features
- Expected: Additional 25-30% reduction (total 40-45%)
- A/B test against baseline

**Phase 3 (Months 5-6):** Advanced embeddings
- Node2Vec + GraphSAGE integration
- Expected: Additional 10-15% reduction (total 50-60%)
- Focus on identity ring detection

**ROI Calculation:**
- Assume $50M annual fraud loss → $25-30M saved annually
- Infrastructure cost: ~$500K/year
- **ROI: 50-60x in first year**

---

## 🔹 For CEO Interviewers

### Q1.7: At a high level, why should we invest in ML-based fraud detection versus just buying a third-party solution?

**A:** 

**Build vs Buy Analysis:**

| Factor | Build (In-House) | Buy (Third-Party) |
|--------|------------------|-------------------|
| **Identity ring detection** | Custom graph embeddings for our data | Generic rules, limited graph |
| **Data privacy** | PII stays internal | Data leaves premises |
| **Cost (Year 1)** | ~$500K | ~$2M+ licensing |
| **Cost (Year 3)** | ~$300K/year | ~$2M+/year |
| **Customization** | Full control over features | Black box |
| **Time to value** | 3-6 months | 1-2 months |
| **Competitive moat** | Proprietary models | Same as competitors |

**Recommendation:** Hybrid approach — buy for quick wins (device fingerprinting, IP reputation), build for differentiation (identity rings, behavioral embeddings). The in-house system becomes a competitive advantage that third-party vendors cannot replicate because they don't have our graph of customer relationships.

---

### Q1.8: What are the top 3 risks to this project, and how do you mitigate them?

**A:**

| Risk | Impact | Mitigation |
|------|--------|------------|
| **1. Model bias causing regulatory action** | Fines, reputational damage | Fairness monitoring across demographics; human-in-the-loop review; regular bias audits |
| **2. False positive spike damaging customer trust** | Churn, revenue loss | Two-stage scoring; progressive friction; real-time monitoring with auto-rollback |
| **3. Fraudsters adapting faster than model retraining** | Evasion, rising losses | Unsupervised guardrail (Isolation Forest); continuous learning pipeline; threat intel integration |

---

# 2. DATA INGESTION & QUALITY

## 🔹 For Technical Interviewers

### Q2.1: The project mentions "messy banking data." What specific messiness do you handle, and how?

**A:**

| Messiness Type | Example | Solution |
|----------------|---------|----------|
| **Inconsistent schemas** | JSON fields vary across sources | Schema enforcement with StructType + mergeSchema option |
| **Malformed identifiers** | Phone: "(555) 123-4567" vs "5551234567" | Regex normalization + standardization pipeline |
| **Duplicate identities** | "John Smith" vs "Jon Smith" at same address | Fuzzy matching (Levenshtein) + entity resolution |
| **Synthetic identities** | Real SSN + fake name | Cross-reference with credit bureaus; graph clustering |
| **High cardinality** | 10M+ device IDs | Hashing trick (modulo 2^20) + target encoding |
| **Time skew** | Events arrive out of order | Watermarking in Structured Streaming; event-time processing |

```python
# Example: Schema enforcement with nullable flags
from pyspark.sql.types import *

transaction_schema = StructType([
    StructField("transaction_id", StringType(), False),  # Non-nullable
    StructField("account_id", StringType(), False),
    StructField("amount", DoubleType(), False),
    StructField("merchant_id", StringType(), True),      # Nullable
    StructField("device_id", StringType(), True),
    StructField("event_time", TimestampType(), False),
    StructField("event_date", DateType(), False)           # Partition key
])

raw = spark.read.schema(transaction_schema).json("/data/raw/*")
```

---

### Q2.2: How do you prevent data leakage between training and validation sets?

**A:** 

**The cardinal rule:** Never use future information to predict the past.

```python
# Time-based split — STRICT enforcement
# Training: 2023-01-01 to 2023-09-30
# Validation: 2023-10-01 to 2023-11-30
# Test: 2023-12-01 to 2023-12-31

train_df = df.filter(col("event_date") <= "2023-09-30")
val_df = df.filter((col("event_date") >= "2023-10-01") & 
                     (col("event_date") <= "2023-11-30"))
test_df = df.filter(col("event_date") >= "2023-12-01")

# Feature cutoff: All features computed ONLY from data before event_date
# Label window: Fraud confirmed within 30 days AFTER transaction
```

**Additional safeguards:**
- **Feature timestamping:** Every feature carries a computed_at timestamp
- **Pipeline validation:** Automated check that no feature computed_at > transaction event_time
- **Holdout accounts:** Some accounts never seen in training (test generalization)
- **Rolling window CV:** Train [Jan-Mar], validate [Apr]; Train [Feb-Apr], validate [May]

---

### Q2.3: How do you handle late-arriving events in a streaming fraud detection context?

**A:**

```python
from pyspark.sql.functions import window

# Structured Streaming with watermarking
stream_df = (spark.readStream
    .format("kafka")
    .option("subscribe", "transactions")
    .load()
    .select(from_json(col("value").cast("string"), schema).alias("data"))
    .select("data.*")
    # Allow 2-hour late events
    .withWatermark("event_time", "2 hours")
    .groupBy(
        window(col("event_time"), "1 hour"),
        col("account_id")
    )
    .agg(sum("amount").alias("hourly_volume"))
)
```

**Trade-off:** Longer watermark = more accurate aggregates but higher latency. For fraud, we use:
- **Short watermark (15 min)** for real-time scoring (approximate features acceptable)
- **Long watermark (24 hr)** for batch feature backfill (exact features for training)
- **Lambda architecture:** Real-time path uses approximate; batch corrects historical

---

### Q2.4: How do you implement idempotent writes to prevent duplicate data?

**A:**

```python
# Delta Lake approach — automatic deduplication
clean.write \
    .format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "true") \
    .partitionBy("event_date") \
    .option("replaceWhere", "event_date = '2024-01-15'") \
    .save("/data/bronze/transactions")

# Or MERGE for upserts
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "/data/bronze/transactions")

(delta_table.alias("target")
    .merge(clean.alias("source"), "target.transaction_id = source.transaction_id")
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute()
)
```

**Why idempotent:** If a job fails and restarts, reprocessing the same files doesn't create duplicates. Critical for financial data where duplicates could double-count fraud or revenue.

---

## 🔹 For Executive Interviewers

### Q2.5: How do you ensure data quality doesn't degrade over time as sources change?

**A:** 

**Data Contract Framework:**
1. **Schema registry:** Centralized schema definitions with versioning
2. **Automated validation:** Great Expectations checks on every batch
3. **SLA monitoring:** Track completeness, freshness, schema drift
4. **Alerting:** PagerDuty if null rate > threshold or volume drops unexpectedly

**Business impact:** Poor data quality directly increases false positives (blocking good customers) and false negatives (missing fraud). We measure data quality as a first-class metric alongside model performance.

---

# 3. EDA & VISUALIZATION

## 🔹 For Technical Interviewers

### Q3.1: What EDA steps are critical for fraud detection, and what visualizations do you produce?

**A:**

**Critical EDA Steps:**

| Step | Purpose | Tool |
|------|---------|------|
| 1. Missingness analysis | Identify data quality issues | df.select([count(when(col(c).isNull(), c)) for c in df.columns]) |
| 2. Label distribution | Confirm class imbalance (typically 0.1-1% fraud) | df.groupBy("is_fraud").count() |
| 3. Time-series fraud rate | Detect seasonality, emerging trends | Matplotlib/Plotly line chart |
| 4. Amount distribution | Identify amount-based fraud patterns | Log-scale histogram |
| 5. Device reuse heatmap | Surface identity rings | Seaborn heatmap |
| 6. Graph degree distribution | Identify hub accounts in fraud networks | NetworkX + Matplotlib |

```python
# PySpark aggregation → Pandas for plotting
fraud_by_day = (df.groupBy("event_date")
    .agg(
        avg(col("is_fraud")).alias("fraud_rate"),
        count("*").alias("volume")
    )
    .orderBy("event_date")
    .toPandas())  # Sample for plotting

import plotly.express as px
fig = px.line(fraud_by_day, x="event_date", y="fraud_rate", 
              title="Fraud Rate Time Series")
fig.show()
```

---

### Q3.2: How do you sample data for EDA without losing rare fraud cases?

**A:** **Stratified sampling** — preserve fraud ratio:

```python
# Stratified sample: 10% of legitimate, 100% of fraud
fraud_df = df.filter(col("is_fraud") == 1)
legit_df = df.filter(col("is_fraud") == 0).sample(fraction=0.1, seed=42)

sample_df = fraud_df.union(legit_df)
# Sample size: ~10% of total, but 100% of fraud cases preserved
```

**For graph analysis:** Use **snowball sampling** — start from known fraud seeds, follow edges to find connected components. This ensures we capture ring structures that random sampling would miss.

---

### Q3.3: What patterns in the data specifically indicate coordinated identity rings?

**A:**

| Pattern | Description | Detection Method |
|---------|-------------|------------------|
| **Shared devices** | 50+ accounts using same device_id | Graph degree centrality |
| **IP clustering** | Accounts from same IP subnet | Geo + network analysis |
| **Synthetic identities** | Similar names, sequential SSNs | Fuzzy matching + sequence detection |
| **Mule accounts** | Rapid money movement between accounts | Transaction graph + velocity |
| **Time coordination** | Identical login timestamps | Behavioral clustering |
| **Attribute mutation** | Same person, slight name variations | Entity resolution + Levenshtein distance |

**Visualization:** Network graph with accounts as nodes, shared attributes as edges. Fraud rings appear as dense subgraphs with high clustering coefficient.

---

## 🔹 For Executive Interviewers

### Q3.4: What did EDA reveal that changed your approach to the problem?

**A:** 

**Key insight from EDA:** 60% of fraud losses came from just 3% of fraud cases — organized rings with 20+ accounts. This shifted our strategy from "detect individual fraud" to "detect fraud networks."

**Before EDA:** Plan was standard supervised classification.
**After EDA:** Added graph embedding layer (Node2Vec) specifically to catch rings. This became the primary differentiator of our solution.

**Business impact:** Graph-based features improved recall on ring fraud from 12% to 78% — a 6.5x improvement on the highest-loss segment.

---

# 4. FEATURE ENGINEERING & EMBEDDINGS

## 🔹 For Technical Interviewers

### Q4.1: What feature categories did you engineer, and which were most predictive?

**A:**

| Category | Examples | Predictive Rank |
|----------|----------|-----------------|
| **Transactional aggregates** | 7/30/90-day sum, count, std of amounts | #3 |
| **Behavioral features** | Login frequency, time-of-day entropy, velocity | #2 |
| **Device/network** | Device reuse count, IP geolocation entropy | #4 |
| **Identity similarity** | Fuzzy match scores, email/phone fingerprints | #5 |
| **Graph features** | Degree, PageRank, connected component size | #1 |
| **Embeddings** | Node2Vec vectors, behavioral sequence embeddings | #1 (combined) |

**Top feature:** Graph-based PageRank score — accounts in fraud rings had 10x higher PageRank than legitimate accounts because they act as hubs connecting many synthetic identities.

---

### Q4.2: Explain how you generate person/entity embeddings. Why are they better than raw features?

**A:**

**Three embedding approaches:**

```python
# 1. Behavioral Sequence Embeddings (LSTM/Transformer)
# Input: Ordered transaction sequence per account
# Output: 128-dim vector capturing spending behavior pattern

from pyspark.ml.feature import Tokenizer
# ... sequence preprocessing ...
# Feed into LSTM → extract hidden state as embedding

# 2. Graph Embeddings (Node2Vec)
# Build bipartite graph: accounts ↔ devices/phones/emails
# Random walks → Skip-gram → 64-dim node vectors

from graphframes import GraphFrame
# Create edges: account_id → device_id
edges = df.select(col("account_id").alias("src"), 
                  col("device_id").alias("dst"))
vertices = df.select("account_id").union(df.select("device_id")).distinct()
vertices = vertices.withColumnRenamed("account_id", "id")

g = GraphFrame(vertices, edges)
# Run Node2Vec via GraphFrames or external library

# 3. Hybrid: Concatenate + Autoencoder dimensionality reduction
behavioral_emb = ...  # 128-dim
graph_emb = ...       # 64-dim
combined = concat(behavioral_emb, graph_emb)  # 192-dim
# Autoencoder → 128-dim final embedding
```

**Why embeddings > raw features:**
- **Generalization:** Embeddings capture latent patterns (e.g., "this account behaves like known fraudsters") even if exact features differ
- **Transfer learning:** Pre-trained embeddings from graph can be reused across models
- **Ring detection:** Accounts in same ring have similar embeddings even if direct features differ (e.g., different names, same behavior)
- **Dimensionality:** 128-dim embedding vs 500+ raw features = faster inference, less overfitting

---

### Q4.3: How do you handle high-cardinality categorical features like merchant_id or device_id?

**A:**

| Technique | When to Use | Implementation |
|-----------|-------------|----------------|
| **Hashing trick** | 10M+ categories, no target leakage risk | hash(merchant_id) % 10000 |
| **Target encoding** | Moderate cardinality, enough data | mean(is_fraud) per merchant_id with smoothing |
| **Embedding layers** | Deep learning models | Learnable embedding matrix in NN |
| **Frequency encoding** | When rare categories matter | count(*) per merchant_id |

**Critical — Preventing target leakage in target encoding:**
```python
# WRONG: Uses global mean (leaks future information)
# RIGHT: Uses only past data per account
from pyspark.sql.window import Window

window_spec = Window.partitionBy("merchant_id") \
    .orderBy("event_time") \
    .rangeBetween(Window.unboundedPreceding, -1)  # Only past data

df = df.withColumn("merchant_fraud_rate", 
    avg("is_fraud").over(window_spec))
```

---

### Q4.4: How do you compute graph features at scale with PySpark?

**A:**

```python
from graphframes import GraphFrame
from pyspark.sql.functions import count, col

# Step 1: Build bipartite graph
account_device = df.select(
    col("account_id").alias("src"),
    col("device_id").alias("dst")
).distinct()

account_phone = df.select(
    col("account_id").alias("src"),
    col("phone_hash").alias("dst")
).distinct()

all_edges = account_device.union(account_phone)
vertices = (all_edges.select(col("src").alias("id"))
    .union(all_edges.select(col("dst").alias("id")))
    .distinct())

g = GraphFrame(vertices, all_edges)

# Step 2: Compute connected components (rings)
sc.setCheckpointDir("/tmp/checkpoints")  # Required for large graphs
components = g.connectedComponents()

# Step 3: Compute degree (how many connections per account)
in_degree = g.inDegrees.withColumnRenamed("inDegree", "device_in_degree")
out_degree = g.outDegrees.withColumnRenamed("outDegree", "device_out_degree")

# Step 4: Join back to features
features = df.join(components, df.account_id == components.id, "left") \
    .join(in_degree, "account_id", "left") \
    .fillna(0, subset=["device_in_degree"])
```

**Performance note:** Connected components on 100M+ nodes requires checkpointing and may take hours. We compute offline (daily batch) and store in feature store, not real-time.

---

## 🔹 For Executive Interviewers

### Q4.5: What is the business value of "embeddings" versus traditional features?

**A:** 

**Analogy:** Traditional features are like checking someone's ID at the door. Embeddings are like recognizing their "vibe" — how they walk, who they're with, where they go.

**Business value:**
- **Catch new fraud types:** Embeddings generalize to unseen patterns. When fraudsters change tactics (new device farms, different names), embeddings still flag them because their "behavioral signature" matches known fraud.
- **Reduce false positives:** A legitimate customer with unusual features (e.g., first large purchase) might be flagged by rules. Embeddings see their long-term behavior is normal → approve.
- **Ring detection:** Raw features can't see that 50 accounts are connected. Embeddings encode graph structure → detect rings that would otherwise steal $500K+ before manual review catches them.

**ROI:** Embeddings alone improved detection rate by 18% while reducing false positives by 12% in our validation.

---

# 5. MODEL SELECTION & TRAINING

## 🔹 For Technical Interviewers

### Q5.1: Walk me through your model selection process. Why LightGBM as primary?

**A:**

**Model progression:**

```
Stage 1: Logistic Regression (Baseline)
    Purpose: Establish minimum viable performance
    Features: Hand-engineered aggregates
    PR-AUC: 0.42

Stage 2: LightGBM (Primary Production)
    Purpose: Best accuracy-speed trade-off for tabular data
    Features: All engineered + graph features
    PR-AUC: 0.71
    Latency: 15ms p99

Stage 3: GraphSAGE + LightGBM (Advanced)
    Purpose: Leverage graph structure for rings
    Features: Node2Vec embeddings + raw features
    PR-AUC: 0.78
    Latency: 45ms p99

Stage 4: Transformer (Experimental)
    Purpose: Sequence modeling for behavioral patterns
    Features: Transaction sequences
    PR-AUC: 0.75
    Latency: 120ms p99 (too slow for real-time)
```

**Why LightGBM won:**
- **Accuracy:** Best PR-AUC on tabular fraud data (industry standard)
- **Speed:** 15ms inference vs 120ms for Transformer
- **Handles imbalance:** Built-in class_weight and is_unbalance parameters
- **Interpretability:** Native feature importance + SHAP support
- **Scalability:** Distributed training via Spark

---

### Q5.2: How do you handle extreme class imbalance (e.g., 0.1% fraud rate)?

**A:**

**Multi-pronged approach:**

```python
import lightgbm as lgb

# 1. Class weights (inverse frequency)
scale_pos_weight = len(negative) / len(positive)  # ~1000:1

# 2. Focal Loss (down-weights easy negatives)
# Custom objective in LightGBM or PyTorch

# 3. Stratified sampling for training
# Keep all fraud, sample negatives at 10:1 ratio

# 4. Evaluation metric: PR-AUC (not ROC-AUC)
# PR-AUC is sensitive to minority class performance

params = {
    'objective': 'binary',
    'metric': 'aucpr',  # PR-AUC
    'scale_pos_weight': scale_pos_weight,
    'is_unbalance': True,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

model = lgb.train(params, train_data, valid_sets=[val_data], 
                  num_boost_round=1000, callbacks=[lgb.early_stopping(50)])
```

**Why not SMOTE?** SMOTE creates synthetic fraud cases that may not be realistic. In fraud, synthetic data can create impossible patterns (e.g., transaction at 3am in NY and 3:05am in LA). We prefer class weights + focal loss + proper evaluation metrics.

---

### Q5.3: When would you use a Graph Neural Network (GNN) versus LightGBM?

**A:**

| Scenario | Model | Reason |
|----------|-------|--------|
| **Individual transaction fraud** | LightGBM | Fast, interpretable, sufficient with good features |
| **Identity ring detection** | GraphSAGE | Graph structure IS the signal; needs message passing |
| **Novel fraud patterns** | GNN + unsupervised | GNN generalizes to unseen graph structures |
| **Real-time constraints (<20ms)** | LightGBM | GNN inference requires graph lookups (slow) |
| **Explainability required** | LightGBM | SHAP works out-of-box; GNN explanations are research-grade |

**Our hybrid approach:**
- **Stage 1 (real-time):** LightGBM scores transaction in <20ms
- **Stage 2 (batch):** GraphSAGE re-scores accounts nightly; flags rings
- **Stage 3 (manual review):** GNN explanations for investigators

---

### Q5.4: How do you train deep learning models (LSTM/Transformer) on PySpark data?

**A:**

```python
# Step 1: Preprocess in PySpark, convert to Petastorm for DL
from petastorm.spark import SparkDatasetConverter, make_spark_converter

spark_df = ...  # PySpark DataFrame with sequences

# Save to parquet for Petastorm
converter = make_spark_converter(spark_df)

# Step 2: PyTorch training with Petastorm loader
import torch
from torch.utils.data import DataLoader
from petastorm.pytorch import DataLoader as PetastormDataLoader

with converter.make_torch_dataloader() as train_loader:
    model = TransformerFraudModel(d_model=128, nhead=8, num_layers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for batch in train_loader:
        # batch: {transaction_seq, amount_seq, time_seq, label}
        output = model(batch['transaction_seq'], batch['amount_seq'])
        loss = focal_loss(output, batch['label'])
        loss.backward()
        optimizer.step()
```

**Key challenge:** PySpark DataFrames are row-based; Transformers need padded sequences. We use sequence_length=50 transactions, pad shorter sequences, truncate longer ones.

---

## 🔹 For Executive Interviewers

### Q5.5: Why not just use a single "best" model? Why multiple models?

**A:** 

**Defense in depth** — no single model catches everything:

| Model | Catches | Misses |
|-------|---------|--------|
| **LightGBM** | Pattern-based fraud (amount, velocity) | New patterns, rings |
| **GraphSAGE** | Identity rings, collusion | Individual opportunistic fraud |
| **Isolation Forest** | Novel anomalies | Known patterns (worse than supervised) |
| **Ensemble** | 90%+ of both | None (covered by other models) |

**Business analogy:** A bank doesn't have one security guard — it has cameras, alarms, guards, and vaults. Each catches different threats.

**Cost:** Running 3 models adds ~$50K/year in compute but prevents $5M+ in fraud that any single model would miss.

---

# 6. EVALUATION & BUSINESS METRICS

## 🔹 For Technical Interviewers

### Q6.1: Why is PR-AUC preferred over ROC-AUC for fraud detection?

**A:**

**ROC-AUC problem:** With 0.1% fraud rate, a model that predicts "all legitimate" gets 99.9% ROC-AUC but catches 0% fraud.

**PR-AUC (Precision-Recall AUC)** focuses on the minority class:
- **Precision:** Of transactions we block, how many are actually fraud?
- **Recall:** Of all fraud, how much do we catch?

```python
from sklearn.metrics import average_precision_score, roc_auc_score

y_true = ...  # Ground truth
y_scores = model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_true, y_scores)      # 0.98 (misleadingly high)
pr_auc = average_precision_score(y_true, y_scores)  # 0.71 (honest)

# PR-AUC is the honest metric for imbalanced data
```

**When to use ROC-AUC:** Model selection during development (ranking models). **When to use PR-AUC:** Final evaluation and production monitoring.

---

### Q6.2: How do you calculate and optimize the business cost metric?

**A:**

**Cost formula:**

```
Total Cost = (FN × L_FN) + (FP × C_FP) + (TP × C_review) + (TN × 0)
```

Where:
- **L_FN** = Average loss per missed fraud = $2,500 (chargeback + operational cost)
- **C_FP** = Cost per false positive = $15 (customer service + friction + potential churn)
- **C_review** = Cost per manual review = $8 (analyst time)

```python
import numpy as np

def business_cost(y_true, y_pred, threshold):
    L_FN = 2500   # Cost of missed fraud
    C_FP = 15     # Cost of false alarm
    C_review = 8  # Cost of manual review
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    
    cost = (fn * L_FN) + (fp * C_FP) + (tp * C_review)
    return cost

# Find optimal threshold
thresholds = np.linspace(0.01, 0.99, 100)
costs = [business_cost(y_val, y_proba, t) for t in thresholds]
optimal_threshold = thresholds[np.argmin(costs)]
# Result: threshold = 0.23 (not 0.5!)
```

**Why this matters:** A threshold of 0.5 (default) might minimize F1 but cost $500K/month. Optimizing for business cost might use threshold 0.23, saving $200K/month while catching more fraud.

---

### Q6.3: Explain your time-based validation strategy. Why not random split?

**A:**

**Random split = data leakage = overoptimistic results = production failure.**

**Time-based split simulates production:**

```
Training:    [Jan 2023]==========[Sep 2023]
Validation:                      [Oct 2023]====[Nov 2023]
Test:                                               [Dec 2023]

Features for Oct transaction: ONLY use data from Jan-Sep
Label for Oct transaction: Fraud confirmed within 30 days of Oct transaction
```

**Rolling window CV (for hyperparameter tuning):**
```
Fold 1: Train [Jan-Mar], Validate [Apr]
Fold 2: Train [Feb-Apr], Validate [May]
Fold 3: Train [Mar-May], Validate [Jun]
```

This ensures the model is evaluated on truly unseen future data, matching production conditions.

---

## 🔹 For CEO/Executive Interviewers

### Q6.4: How do you translate model metrics into business impact that the board understands?

**A:**

**Translation framework:**

| Technical Metric | Business Translation | Board-Level Statement |
|------------------|---------------------|----------------------|
| **PR-AUC: 0.71** | We catch 71% of fraud while keeping false alarms manageable | "Our system stops 7 out of 10 fraud attempts before money leaves the bank" |
| **Recall: 68%** | 68% of fraud transactions blocked | "$17M prevented annually vs $8M lost" |
| **Precision: 12%** | 12% of blocked transactions are actual fraud (rest reviewed) | "For every 100 alerts, 12 are confirmed fraud; 88 are safe customers we inconvenience" |
| **False Positive Rate: 0.5%** | 5 out of 1000 good customers get friction | "We inconvenience 1 in 200 good customers to protect everyone" |

**ROI dashboard:**
```
Monthly Fraud Loss (before ML): $4.2M
Monthly Fraud Loss (after ML):  $2.1M
Monthly Savings:                $2.1M
Monthly Operating Cost:         $45K
Net Monthly Benefit:            $2.055M
Annual ROI:                     ($24.66M savings - $0.54M cost) / $0.54M = 45x
```

---

# 7. HYPERPARAMETER TUNING

## 🔹 For Technical Interviewers

### Q7.1: What hyperparameter tuning strategy do you use for LightGBM?

**A:**

**Bayesian Optimization with Optuna:**

```python
import optuna
from optuna.integration import LightGBMPruningCallback

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'aucpr',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1
    }
    
    model = lgb.train(param, train_data, 
                      valid_sets=[val_data],
                      num_boost_round=1000,
                      callbacks=[LightGBMPruningCallback(trial, 'aucpr'),
                                 lgb.early_stopping(50)])
    
    return model.best_score['valid_0']['aucpr']

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=50, timeout=3600)

print(f"Best params: {study.best_params}")
print(f"Best PR-AUC: {study.best_value}")
```

**Why Bayesian over Grid Search:**
- Grid search wastes trials on bad combinations
- Bayesian learns from previous trials, focuses on promising regions
- 50 Bayesian trials ≈ 500 grid search trials in quality

---

### Q7.2: How do you tune hyperparameters for deep learning models?

**A:**

**Population-based training (PBT) for neural nets:**

```python
# Ray Tune example for Transformer
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

config = {
    "d_model": tune.choice([64, 128, 256]),
    "nhead": tune.choice([4, 8, 16]),
    "num_layers": tune.choice([2, 4, 6]),
    "dropout": tune.uniform(0.1, 0.5),
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([32, 64, 128])
}

pbt_scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=5,
    hyperparam_mutations={
        "lr": lambda: np.random.uniform(1e-5, 1e-3),
        "batch_size": lambda: np.random.choice([32, 64, 128])
    }
)

analysis = tune.run(
    train_transformer,
    config=config,
    scheduler=pbt_scheduler,
    num_samples=30,
    resources_per_trial={"gpu": 1},
    metric="val_pr_auc",
    mode="max"
)
```

**Key difference from LightGBM:** Neural nets need more trials (30-50) and benefit from PBT's ability to transfer weights between configurations.

---

### Q7.3: How do you prevent overfitting during hyperparameter tuning?

**A:**

**Multi-layered defense:**

1. **Nested cross-validation:** Outer loop for evaluation, inner loop for tuning
2. **Early stopping:** Stop training when validation metric plateaus
3. **Regularization in search space:** Include L1/L2, dropout as tunable params
4. **Holdout test set:** Never touch during tuning; final evaluation only
5. **Complexity penalty:** Prefer simpler models if performance within 1%

```python
# Nested CV example
from sklearn.model_selection import KFold

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

for train_idx, test_idx in outer_cv.split(X):
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    
    # Inner loop: tune on X_train_outer
    best_params = tune_on_fold(X_train_outer, y[train_idx], inner_cv)
    
    # Evaluate on held-out test
    model = LightGBM(**best_params)
    model.fit(X_train_outer, y[train_idx])
    score = evaluate(model, X_test_outer, y[test_idx])
    
    outer_scores.append(score)

final_score = np.mean(outer_scores)  # Honest estimate
```

---

## 🔹 For Executive Interviewers

### Q7.4: How long does hyperparameter tuning take, and is it worth the investment?

**A:**

**Time investment:**
- **LightGBM:** 2-4 hours on 8-core machine (50 trials)
- **Transformer:** 12-24 hours on 4-GPU cluster (30 trials)
- **GraphSAGE:** 6-12 hours on GPU (40 trials)

**ROI calculation:**
- Engineer time: 8 hours @ $150/hr = $1,200
- Compute cost: $200 (cloud GPUs)
- **Total: $1,400 per tuning run**

**Benefit:**
- Untuned model PR-AUC: 0.65
- Tuned model PR-AUC: 0.71
- **6% improvement = $1.5M additional fraud prevented annually**

**Verdict:** 1000x ROI on tuning investment. We tune quarterly as new data arrives.

---

# 8. EXPLAINABILITY & FAIRNESS

## 🔹 For Technical Interviewers

### Q8.1: How do you explain model predictions to fraud investigators who need actionable reasons?

**A:**

**SHAP (SHapley Additive exPlanations) for local explanations:**

```python
import shap

# Train explainer on background sample
explainer = shap.TreeExplainer(model, X_train_sample)
shap_values = explainer.shap_values(X_val_sample)

# For a specific prediction
prediction_idx = 42
shap.force_plot(explainer.expected_value, shap_values[0][prediction_idx], 
                X_val_sample.iloc[prediction_idx], 
                matplotlib=True)

# Top features for this prediction
top_features = np.argsort(np.abs(shap_values[0][prediction_idx]))[-5:]
for feat in top_features:
    print(f"{X_val.columns[feat]}: {shap_values[0][prediction_idx][feat]:.3f}")
```

**Human-readable output for investigators:**
```
Transaction ID: TXN-2024-001234
Risk Score: 0.87 (HIGH RISK)

Top 5 Reasons:
1. ⚠️ Device used by 47 other accounts in past 7 days (+0.34 risk)
2. ⚠️ Transaction amount 8x higher than 30-day average (+0.22 risk)
3. ⚠️ IP location 500 miles from home address (+0.15 risk)
4. ⚠️ Account created 3 days ago (+0.11 risk)
5. ⚠️ 5 failed login attempts in past hour (+0.05 risk)

Recommended Action: Block transaction, request identity verification
```

---

### Q8.2: How do you ensure the model doesn't discriminate based on protected characteristics?

**A:**

**Fairness audit pipeline:**

```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# Map protected attributes (must be done carefully to avoid leakage)
protected_attrs = ['age_group', 'gender', 'zip_code_income_quintile']

# Compute disparate impact
di = (
    favorable_rate_protected / favorable_rate_unprotected
)

# Acceptable range: 0.8 ≤ DI ≤ 1.25 (80% rule)
if di < 0.8:
    print(f"⚠️ Disparate impact detected: {di:.2f}")
    # Mitigation: reweighting, adversarial debiasing, or threshold adjustment

# Equal opportunity difference
eod = (
    true_positive_rate_protected - true_positive_rate_unprotected
)

# Acceptable: |EOD| < 0.1
```

**Mitigation strategies:**
1. **Pre-processing:** Reweight samples to balance outcomes
2. **In-processing:** Adversarial training to remove protected attribute signal
3. **Post-processing:** Adjust thresholds per group to equalize TPR

**Legal compliance:** Document all fairness metrics for ECOA (Equal Credit Opportunity Act) and FCRA (Fair Credit Reporting Act) examinations.

---

### Q8.3: How do you handle "adverse action" notices required by regulation?

**A:**

**Automated adverse action letter generation:**

```python
def generate_adverse_action_notice(customer_id, shap_values, feature_names):
    """
    Generate FCRA-compliant adverse action notice with top reasons
    """
    top_reasons = get_top_negative_features(shap_values, n=4)
    
    notice = f"""
    ADVERSE ACTION NOTICE
    
    Dear Customer,
    
    We have taken an adverse action regarding your application/transaction.
    
    Principal reasons for this action:
    1. {top_reasons[0]}
    2. {top_reasons[1]}
    3. {top_reasons[2]}
    4. {top_reasons[3]}
    
    You have the right to request a copy of your credit report and to dispute
    any inaccurate information.
    
    Contact: compliance@bank.com | 1-800-XXX-XXXX
    """
    
    # Store in compliance database
    save_to_compliance_db(customer_id, notice, timestamp=datetime.now())
    
    return notice
```

**Compliance checklist:**
- ✅ Top 4 reasons clearly stated
- ✅ Contact information provided
- ✅ Right to dispute mentioned
- ✅ Stored for 25 months (regulatory requirement)
- ✅ Sent within 30 days (legal deadline)

---

## 🔹 For Executive Interviewers

### Q8.4: What is our liability if the model is biased, and how do we protect the company?

**A:**

**Liability exposure:**
- **ECOA violations:** Up to $10K per violation + punitive damages
- **Class action lawsuits:** Potential $10M+ settlements (see Apple Card gender bias case)
- **Reputational damage:** Customer churn, media scrutiny

**Protection framework:**
1. **Model governance board:** Quarterly reviews with Legal, Compliance, Risk
2. **Bias testing:** Automated fairness metrics on every model deployment
3. **Human-in-the-loop:** High-risk decisions (large transactions, new accounts) require manual review
4. **Documentation:** Complete model cards with fairness metrics, limitations, intended use
5. **Insurance:** E&O (Errors & Omissions) policy covering AI/ML decisions

**Recent precedent:** Goldman Sachs investigated for Apple Card algorithm (2019). No formal charges, but reputational damage and costly investigation. **Lesson:** Proactive fairness monitoring is cheaper than reactive crisis management.

---

# 9. DEPLOYMENT & SERVING

## 🔹 For Technical Interviewers

### Q9.1: Describe your deployment architecture for real-time fraud scoring.

**A:**

**Production architecture:**

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Transaction│────▶│  API Gateway │────▶│  Load       │
│  Initiation │     │  (Kong/AWS)  │     │  Balancer   │
└─────────────┘     └──────────────┘     └─────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
           ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
           │  Model Pod 1 │          │  Model Pod 2 │          │  Model Pod N │
           │  (ONNX)      │          │  (ONNX)      │          │  (ONNX)      │
           └──────────────┘          └──────────────┘          └──────────────┘
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                                     ┌──────────────┐
                                     │  Feature     │
                                     │  Store       │
                                     │  (Redis)     │
                                     └──────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
           ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
           │  Batch       │          │  Monitoring  │          │  Alerting    │
           │  Processor   │          │  (Prometheus)│          │  (PagerDuty) │
           └──────────────┘          └──────────────┘          └──────────────┘
```

**Latency budget (p99 < 50ms):**
- Feature fetch from Redis: 5ms
- Model inference (ONNX): 10ms
- Network overhead: 5ms
- **Buffer:** 30ms for spikes

**Tech stack:**
- **Serving:** FastAPI + Uvicorn (async)
- **Model format:** ONNX (optimized, cross-platform)
- **Feature store:** Redis (low-latency) + Feast (metadata)
- **Orchestration:** Kubernetes (auto-scaling)
- **CI/CD:** GitHub Actions + ArgoCD

---

### Q9.2: How do you prevent training-serving skew?

**A:**

**Skew prevention strategies:**

1. **Unified feature pipeline:**
```python
# Same code for training and serving
class FeaturePipeline:
    def compute_features(self, df):
        # Window functions, aggregations, encodings
        # Used identically in training and serving
        pass

# Training
train_features = FeaturePipeline().compute_features(train_df)

# Serving (real-time)
realtime_features = FeaturePipeline().compute_features(latest_df)
```

2. **Feature store with point-in-time correctness:**
```python
from feast import FeatureStore

store = FeatureStore(repo_path="./feature_repo")

# Training: historical features with correct as-of timestamps
training_df = store.get_historical_features(
    entity_df=transactions_with_timestamps,
    features=[
        "account_features:avg_amount_30d",
        "account_features:device_reuse_count",
        "graph_features:page_rank"
    ]
).to_df()

# Serving: latest features
realtime_features = store.get_online_features(
    entity_rows=[{"account_id": "ACC-12345"}],
    features=[...]
).to_dict()
```

3. **Automated skew detection:**
```python
# Daily job comparing training vs serving distributions
def detect_skew():
    training_stats = compute_stats(training_features)
    serving_stats = compute_stats(serving_features_last_24h)
    
    psi = calculate_psi(training_stats, serving_stats)
    if psi > 0.2:
        alert("Training-serving skew detected!")
```

---

### Q9.3: How do you implement canary deployment for fraud models?

**A:**

**Canary rollout strategy:**

```yaml
# Kubernetes traffic splitting
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: fraud-model-vs
spec:
  hosts:
  - fraud-model-service
  http:
  - route:
    - destination:
        host: fraud-model-v1
      weight: 90  # 90% to current model
    - destination:
        host: fraud-model-v2
      weight: 10  # 10% to new model (canary)
```

**Monitoring during canary:**
```python
def monitor_canary(canary_percentage=10):
    v1_metrics = get_metrics("fraud-model-v1")
    v2_metrics = get_metrics("fraud-model-v2")
    
    # Check for significant differences
    if abs(v1_metrics.fraud_rate - v2_metrics.fraud_rate) > 0.05:
        rollback()
        alert("Canary showing different fraud rate!")
    
    if v2_metrics.latency_p99 > 100:  # ms
        rollback()
        alert("Canary latency too high!")
    
    # If healthy, increase canary traffic
    if all_checks_pass():
        increase_canary_traffic(10 → 25 → 50 → 100)
```

**Rollback triggers:**
- Fraud rate difference > 5%
- Latency p99 > 100ms
- Error rate > 1%
- False positive spike > 20%

---

### Q9.4: How do you handle model versioning and rollback?

**A:**

**Model registry with MLflow:**

```python
import mlflow

# Register model
mlflow.log_param("model_type", "LightGBM")
mlflow.log_metric("pr_auc", 0.71)
mlflow.sklearn.log_model(model, "model")

# Transition to staging
client = MlflowClient()
client.transition_model_version_stage(
    name="fraud-detection-model",
    version=3,
    stage="staging"
)

# After validation, promote to production
client.transition_model_version_stage(
    name="fraud-detection-model",
    version=3,
    stage="production"
)

# Rollback if needed
client.transition_model_version_stage(
    name="fraud-detection-model",
    version=2,  # Previous version
    stage="production"
)
```

**Rollback playbook:**
1. Detect issue (monitoring alert)
2. Freeze new deployments
3. Rollback to previous version (automated via CI/CD)
4. Investigate root cause
5. Document incident
6. Resume deployments after fix

---

## 🔹 For Executive Interviewers

### Q9.5: What is the uptime SLA for the fraud detection system, and what happens if it goes down?

**A:**

**SLA commitments:**
- **Availability:** 99.95% (≤22 minutes downtime/month)
- **Latency:** p99 < 50ms
- **Accuracy:** PR-AUC ≥ 0.68 (monitored weekly)

**Fallback modes if system fails:**
1. **Degraded mode:** Switch to rule-based heuristics (no ML)
   - Catches ~40% of fraud (vs 70% with ML)
   - Higher false positives but keeps transactions flowing
2. **Manual review queue:** Route high-risk transactions to human analysts
   - Slower (minutes vs milliseconds)
   - Prevents catastrophic fraud losses
3. **Hard decline:** For highest-risk segments (new accounts, large amounts)
   - Last resort; impacts customer experience

**Business continuity:**
- **RTO (Recovery Time Objective):** 15 minutes
- **RPO (Recovery Point Objective):** 5 minutes (max data loss)
- **Redundancy:** Multi-AZ deployment, failover to secondary region

**Cost of downtime:**
- 1 hour outage = ~$500K in fraud losses + reputational damage
- Investment in redundancy ($200K/year) is justified

---

# 10. MONITORING & DRIFT DETECTION

## 🔹 For Technical Interviewers

### Q10.1: What metrics do you monitor in production, and how do you alert on them?

**A:**

**Monitoring dashboard:**

| Metric | Threshold | Alert Severity | Action |
|--------|-----------|----------------|--------|
| **Prediction latency (p99)** | > 50ms | Warning | Scale up pods |
| **Prediction latency (p99)** | > 100ms | Critical | Auto-rollback |
| **Error rate** | > 1% | Warning | Investigate logs |
| **Error rate** | > 5% | Critical | Rollback |
| **Fraud rate** | ±20% from baseline | Warning | Check for drift |
| **False positive rate** | > 2x baseline | Critical | Threshold adjustment |
| **Feature null rate** | > 5% for critical features | Warning | Data quality investigation |
| **PSI (Population Stability Index)** | > 0.2 | Warning | Retraining trigger |

**Alerting implementation:**
```python
from prometheus_client import Counter, Histogram
import pagerduty

# Metrics
prediction_latency = Histogram('fraud_prediction_latency_ms', 'Prediction latency')
fraud_rate = Gauge('fraud_detection_rate', 'Fraud rate')
error_rate = Gauge('fraud_service_error_rate', 'Error rate')

# Alerting logic
def check_alerts():
    if error_rate._value.get() > 0.05:
        pagerduty.trigger_incident(
            service_key="fraud-service",
            description="Error rate exceeded 5%",
            severity="critical"
        )
    
    if psi_score > 0.2:
        pagerduty.trigger_incident(
            service_key="fraud-service",
            description=f"Data drift detected (PSI={psi_score:.2f})",
            severity="warning"
        )
```

---

### Q10.2: How do you detect concept drift vs data drift?

**A:**

**Definitions:**
- **Data drift (covariate shift):** Input feature distributions change (P(X) changes)
- **Concept drift:** Relationship between features and target changes (P(Y|X) changes)

**Detection methods:**

```python
# Data drift: PSI (Population Stability Index)
def calculate_psi(expected, actual, buckets=10):
    """Compare expected vs actual feature distributions"""
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets+1)[1:-1])
    
    expected_buckets = np.digitize(expected, breakpoints)
    actual_buckets = np.digitize(actual, breakpoints)
    
    expected_dist = np.bincount(expected_buckets, minlength=buckets) / len(expected)
    actual_dist = np.bincount(actual_buckets, minlength=buckets) / len(actual)
    
    psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist + 1e-10))
    return psi

# Concept drift: Monitor residual errors over time
def detect_concept_drift(y_true_window, y_pred_window):
    """Check if model performance degrades over time"""
    recent_auc = roc_auc_score(y_true_window[-7:], y_pred_window[-7:])
    baseline_auc = roc_auc_score(y_true_window[:30], y_pred_window[:30])
    
    if baseline_auc - recent_auc > 0.05:
        return True  # Concept drift detected
    return False

# Combined monitoring
if calculate_psi(train_features, serving_features) > 0.2:
    alert("Data drift detected")
    
if detect_concept_drift(y_true_recent, y_pred_recent):
    alert("Concept drift detected - model retraining needed")
```

**Action plan:**
- **Data drift:** Investigate upstream data sources; may not require retraining
- **Concept drift:** Trigger immediate model retraining; fraudsters adapted

---

### Q10.3: How do you handle adversarial attacks where fraudsters try to game the model?

**A:**

**Adversarial robustness strategies:**

1. **Unsupervised guardrail:**
```python
# Isolation Forest as backup
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train)

def predict_with_guardrail(x):
    ml_prediction = model.predict_proba(x)[0, 1]
    iso_score = -iso_forest.score_samples([x])[0]  # Higher = more anomalous
    
    # If ML says "safe" but anomaly detector says "suspicious"
    if ml_prediction < 0.3 and iso_score > 0.7:
        return "MANUAL_REVIEW"  # Escalate
    
    return ml_prediction
```

2. **Adversarial training:**
```python
# Generate adversarial examples during training
def generate_adversarial_examples(X, y, model, epsilon=0.1):
    """Create perturbed examples that fool the model"""
    X_adv = X.copy()
    gradient = compute_gradient(model, X, y)
    X_adv += epsilon * np.sign(gradient)
    return X_adv

# Train on mix of real and adversarial examples
X_train_augmented = np.vstack([X_train, generate_adversarial_examples(X_train, y_train, model)])
y_train_augmented = np.hstack([y_train, y_train])
```

3. **Ensemble diversity:**
- Train multiple models with different architectures
- Fraudsters must evade all models simultaneously (harder)

4. **Threat intelligence integration:**
- Subscribe to fraudster forum monitoring services
- Update blacklists and features based on emerging tactics

---

## 🔹 For Executive Interviewers

### Q10.4: How often do you retrain the model, and what triggers an emergency retrain?

**A:**

**Retraining schedule:**
- **Scheduled:** Weekly incremental, monthly full retrain
- **Trigger-based:** Immediate retrain if drift detected

**Emergency retrain triggers:**
1. **Concept drift:** PR-AUC drops > 0.05 in 7-day window
2. **Fraud spike:** Fraud loss rate doubles week-over-week
3. **New attack pattern:** Threat intel identifies new fraud tactic bypassing model
4. **Regulatory change:** New compliance requirements affect model features

**Emergency retrain playbook:**
```
Hour 0: Detect issue (alert triggered)
Hour 1: Assemble incident response team (DS, Eng, Risk)
Hour 2: Analyze root cause (drift? new pattern? data issue?)
Hour 4: Collect new training data with attack patterns
Hour 8: Train candidate models
Hour 12: Validate models (backtest + stress test)
Hour 16: Deploy via canary (10% traffic)
Hour 20: Monitor canary; if stable, increase to 50%
Hour 24: Full rollout if no issues
```

**Cost:** Emergency retrain = $50K in engineer time + compute. Prevents $2M+ in fraud losses. **ROI: 40x.**

---

# 11. TESTING & VALIDATION

## 🔹 For Technical Interviewers

### Q11.1: What tests do you write for a fraud detection pipeline?

**A:**

**Testing pyramid:**

```
                    ┌─────────────┐
                   ╱│  E2E Tests  │╲
                  ╱ │ (10% coverage)│╲
                 ╱  └─────────────┘  ╲
                ╱──────────────────────╲
               ╱   Integration Tests    ╲
              ╱    (20% coverage)        ╲
             ╱────────────────────────────╲
            ╱      Unit Tests              ╲
           ╱      (70% coverage)            ╲
          ╱──────────────────────────────────╲
```

**Unit tests (pytest):**
```python
def test_feature_pipeline_handles_nulls():
    input_df = spark.createDataFrame([
        ("ACC1", None, 100.0),
        ("ACC2", "DEV1", 200.0)
    ], ["account_id", "device_id", "amount"])
    
    result = compute_features(input_df)
    
    assert result.filter(col("device_reuse_count").isNull()).count() == 0
    assert result.collect()[0]["device_reuse_count"] == 0  # Null → 0

def test_schema_enforcement_rejects_invalid():
    invalid_df = spark.createDataFrame([
        ("TXN1", "ACC1", "not_a_number")  # amount should be float
    ], ["transaction_id", "account_id", "amount"])
    
    with pytest.raises(Exception):
        enforce_schema(invalid_df, transaction_schema)
```

**Integration tests:**
```python
def test_end_to_end_scoring():
    # Simulate real transaction
    transaction = create_test_transaction(amount=5000, is_new_device=True)
    
    # Call scoring service
    response = requests.post("http://localhost:8000/score", json=transaction)
    
    assert response.status_code == 200
    assert "risk_score" in response.json()
    assert 0 <= response.json()["risk_score"] <= 1
    assert response.elapsed.total_seconds() < 0.1  # <100ms
```

**A/B test framework:**
```python
def run_ab_test(control_model, treatment_model, traffic_split=0.5):
    """
    Randomly assign transactions to control/treatment
    Compare fraud detection rate and false positive rate
    """
    results = {
        "control": {"fraud_caught": 0, "false_positives": 0, "total": 0},
        "treatment": {"fraud_caught": 0, "false_positives": 0, "total": 0}
    }
    
    for transaction in live_stream:
        bucket = "treatment" if random() < traffic_split else "control"
        model = treatment_model if bucket == "treatment" else control_model
        
        prediction = model.predict(transaction)
        outcome = get_outcome(transaction.id)  # Wait for label
        
        results[bucket]["total"] += 1
        if prediction == 1 and outcome == 1:
            results[bucket]["fraud_caught"] += 1
        elif prediction == 1 and outcome == 0:
            results[bucket]["false_positives"] += 1
    
    # Statistical significance test
    lift = (results["treatment"]["fraud_caught"] / results["treatment"]["total"]) / \
           (results["control"]["fraud_caught"] / results["control"]["total"])
    
    p_value = bootstrap_significance_test(results)
    
    return {"lift": lift, "p_value": p_value, "results": results}
```

---

### Q11.2: How do you validate model performance before production deployment?

**A:**

**Multi-stage validation:**

```
Stage 1: Backtesting
  - Train on [Jan-Aug], validate on [Sep]
  - Must beat baseline PR-AUC by ≥ 0.02
  - Must meet fairness constraints (DI ≥ 0.8)

Stage 2: Shadow Mode
  - Deploy to production, log predictions, don't act on them
  - Compare predicted fraud rate vs actual (2-week lag)
  - Validate latency under real load

Stage 3: Canary (10% traffic)
  - Act on predictions for 10% of transactions
  - Monitor fraud loss rate, false positive rate
  - Statistical comparison vs control group

Stage 4: Gradual Rollout
  - 10% → 25% → 50% → 100% over 1 week
  - Auto-rollback if metrics degrade
```

**Validation checklist:**
- ✅ PR-AUC ≥ 0.68 on holdout test set
- ✅ Latency p99 < 50ms under load test (10K TPS)
- ✅ Fairness metrics within legal bounds
- ✅ No training-serving skew (PSI < 0.1)
- ✅ Explainability outputs validated by fraud investigators
- ✅ Disaster recovery tested (failover to backup region)

---

### Q11.3: How do you calculate confidence intervals for model metrics?

**A:**

**Bootstrap confidence intervals:**

```python
from sklearn.utils import resample

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci_level=0.95):
    """
    Calculate confidence interval using bootstrap resampling
    """
    bootstrapped_scores = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = resample(range(len(y_true)), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metric
        score = metric_fn(y_true_boot, y_pred_boot)
        bootstrapped_scores.append(score)
    
    # Calculate CI
    alpha = 1 - ci_level
    lower = np.percentile(bootstrapped_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))
    
    return {
        "point_estimate": np.mean(bootstrapped_scores),
        "ci_lower": lower,
        "ci_upper": upper,
        "std": np.std(bootstrapped_scores)
    }

# Example usage
pr_auc_ci = bootstrap_ci(y_val, y_proba, average_precision_score)
print(f"PR-AUC: {pr_auc_ci['point_estimate']:.3f} ({pr_auc_ci['ci_lower']:.3f}-{pr_auc_ci['ci_upper']:.3f})")
# Output: PR-AUC: 0.712 (0.698-0.726)
```

**Why bootstrap:** Fraud data is imbalanced; analytical CI formulas assume normality which doesn't hold. Bootstrap makes no distributional assumptions.

---

## 🔹 For Executive Interviewers

### Q11.4: What is your definition of "done" for a model ready for production?

**A:**

**Production readiness checklist:**

| Category | Requirement | Owner |
|----------|-------------|-------|
| **Performance** | PR-AUC ≥ 0.68, Recall ≥ 65%, Precision ≥ 10% | Data Science |
| **Latency** | p99 < 50ms at 10K TPS | Engineering |
| **Fairness** | Disparate impact ≥ 0.8, EOD < 0.1 | Compliance |
| **Explainability** | SHAP explanations for all predictions | Data Science |
| **Monitoring** | Dashboards, alerts, drift detection configured | MLOps |
| **Documentation** | Model card, runbook, API docs complete | Tech Writing |
| **Security** | Penetration test passed, PII encrypted | Security |
| **Disaster Recovery** | Failover tested, RTO < 15 min | SRE |
| **Legal** | FCRA/ECOA compliance sign-off | Legal |
| **Business** | ROI projection ≥ 10x, stakeholder approval | Product |

**Sign-off required from:** VP Data Science, VP Engineering, Chief Risk Officer, General Counsel.

---

# 12. CEO & EXECUTIVE STRATEGIC QUESTIONS

## 🔹 For CEO Interviewers

### Q12.1: If I gave you 3x the budget, what would you do differently? If I cut your budget by 50%, what would you stop doing?

**A:**

**3x Budget Scenario ($1.5M/year):**
1. **Real-time graph embeddings:** Deploy GraphSAGE for sub-second ring detection (currently batch)
2. **Federated learning:** Partner with other banks to train on combined data without sharing PII
3. **Dedicated fraud research team:** 5 FTEs focused on emerging threats and adversarial ML
4. **Enhanced data sources:** Purchase premium threat intel feeds, dark web monitoring
5. **Global expansion:** Deploy in EU/APAC regions with localized models

**Expected impact:** Fraud loss reduction from 60% → 75%; competitive moat widens

**50% Budget Cut ($250K/year):**
1. ❌ Stop: Advanced embeddings (Node2Vec, GraphSAGE) — revert to hand-crafted graph features
2. ❌ Stop: Real-time serving — move to near-real-time (5-minute batches)
3. ❌ Stop: Continuous retraining — weekly instead of daily
4. ❌ Stop: External threat intel — rely on internal data only
5. ✅ Keep: Core LightGBM model, basic monitoring, compliance

**Expected impact:** Fraud loss reduction from 60% → 40%; increased manual review workload

---

### Q12.2: What is the single biggest competitive advantage this system gives us?

**A:**

**Proprietary identity ring detection.**

**Why it's a moat:**
- **Data network effect:** More customers → larger graph → better ring detection → lower fraud → more customers
- **Impossible to buy:** Third-party vendors don't have our customer graph; they see isolated transactions
- **Compounding advantage:** Each fraud ring we catch improves embeddings for future detection
- **Regulatory barrier:** Competitors can't easily replicate due to data privacy restrictions

**Quantified advantage:**
- Our system catches 78% of ring fraud; industry average is 35%
- Ring fraud = 60% of total fraud losses
- **Competitive edge:** $15M/year lower fraud losses vs competitors with same transaction volume

**Strategic implication:** This isn't just a cost center — it's a product differentiator we can market to enterprise customers ("Bank with us; we protect you better").

---

### Q12.3: How do you think about the trade-off between fraud prevention and customer experience?

**A:**

**Framework: Progressive Friction Model**

```
Risk Score Range | Customer Experience | Friction Level
-----------------|---------------------|---------------
0.00 - 0.15      | Zero friction       | Approve instantly
0.15 - 0.30      | Light friction      | SMS verification
0.30 - 0.50      | Medium friction     | Knowledge-based questions
0.50 - 0.70      | High friction       | Document upload + manual review
0.70 - 1.00      | Blocked             | Decline transaction
```

**Philosophy:**
- **Good customers should never notice us:** 95% of transactions are zero-friction
- **Bad actors should feel the friction:** Escalating hurdles make fraud unprofitable
- **Transparency:** Customers understand why we're asking for verification (security, not distrust)

**Metrics:**
- **Friction rate:** Target < 5% of transactions (currently 3.8%)
- **Friction conversion:** 85% of verified transactions complete successfully
- **NPS impact:** Customers who pass verification show +2 NPS points (feel protected)

**CEO soundbite:** "We're not in the business of saying 'no' — we're in the business of saying 'yes' safely."

---

### Q12.4: What keeps you up at night about this system?

**A:**

**Top 3 concerns:**

1. **Black swan fraud event:**
   - Scenario: Coordinated attack exploiting unknown vulnerability
   - Impact: $10M+ loss in 48 hours
   - Mitigation: Circuit breakers (auto-decline if loss rate spikes 5x), insurance

2. **Algorithmic bias scandal:**
   - Scenario: Media exposes disproportionate false positives for certain demographics
   - Impact: Reputational damage, regulatory investigation, class action
   - Mitigation: Proactive fairness audits, diverse review board, transparency reports

3. **Talent retention:**
   - Scenario: Key ML engineers recruited by Big Tech
   - Impact: Institutional knowledge loss, slower innovation
   - Mitigation: Competitive comp, equity, mission-driven culture, succession planning

**Sleep aid:** Weekly risk review with CRO, monthly board updates, $5M cyber/fraud insurance policy.

---

### Q12.5: Where do you see this system in 3 years?

**A:**

**Vision: Autonomous Fraud Prevention Platform**

**Year 1 (Current state):**
- Supervised ML + graph features
- Human-in-the-loop for edge cases
- 60% fraud loss reduction

**Year 2 (Near-term):**
- Self-supervised learning (less labeled data dependency)
- Real-time graph neural networks
- Cross-channel fraud detection (banking + payments + lending)
- 70% fraud loss reduction

**Year 3 (Target state):**
- Federated learning consortium with 5+ partner banks
- Generative AI for synthetic fraud scenario simulation
- Fully automated for 99% of transactions (human review for 1%)
- 80% fraud loss reduction
- **Product spin-off:** Offer fraud detection as SaaS to smaller banks

**Strategic outcome:** Transform from cost center to revenue generator ($50M ARR from SaaS).

---

# 13. STAKEHOLDER & PRODUCT QUESTIONS

## 🔹 For Product/Growth Interviewers

### Q13.1: How do you work with Product to balance fraud prevention vs onboarding conversion?

**A:**

**Collaboration framework:**

1. **Shared metrics:**
   - **North Star:** Net Revenue Retention (fraud losses - friction costs + approved volume)
   - Not just "fraud rate" (Risk) or "conversion rate" (Product)

2. **Joint experimentation:**
   ```
   Experiment: Relax threshold for transactions < $500
   
   Risk concern: +$50K fraud loss/month
   Product benefit: +2% conversion = +$200K revenue/month
   
   Net impact: +$150K/month → Ship it
   ```

3. **Progressive onboarding:**
   - Day 1: $1K daily limit (low risk)
   - Day 7: $5K limit if no fraud signals
   - Day 30: $25K limit with established behavior

4. **Escalation protocol:**
   - Disagreement → A/B test → Data decides
   - Emergency: CRO has veto on risk, CPO has veto on UX

**Example win:** Product wanted to remove SMS verification for low-risk users. Risk objected. We A/B tested: 0.02% increase in fraud vs 3% conversion lift. Shipped with monitoring. Net +$500K/year.

---

### Q13.2: How do you prioritize the fraud detection backlog?

**A:**

**Prioritization matrix:**

| Impact | Effort | Priority | Example |
|--------|--------|----------|---------|
| **High** | **Low** | P0 (Do now) | Fix label leakage bug |
| **High** | **High** | P1 (Plan next quarter) | Real-time graph embeddings |
| **Low** | **Low** | P2 (Backlog) | Dashboard UI improvements |
| **Low** | **High** | P3 (Probably not) | Rewrite in Rust |

**Scoring formula:**
```
Priority Score = (Fraud Loss Prevented × Probability of Success) / Engineering Weeks

Example:
- GraphSAGE upgrade: ($5M × 0.7) / 12 weeks = 0.29
- Threshold optimization: ($500K × 0.95) / 2 weeks = 0.24
→ GraphSAGE wins despite higher effort
```

**Stakeholder input:** Risk, Product, Engineering each get 100 points to allocate quarterly. Forces trade-off discussions.

---

### Q13.3: A major fraud ring just stole $2M. How do you respond in the next 24 hours?

**A:**

**Incident response playbook:**

**Hour 0-2: Containment**
- [ ] Freeze all accounts in identified ring (50-100 accounts)
- [ ] Block associated devices, IPs, phone numbers
- [ ] Notify law enforcement (FBI IC3 report)
- [ ] Activate war room (Risk, Eng, DS, Legal, Comms)

**Hour 2-8: Investigation**
- [ ] Forensic analysis: How did they bypass controls?
- [ ] Identify scope: Any other compromised accounts?
- [ ] Check for insider involvement
- [ ] Preserve evidence for prosecution

**Hour 8-24: Remediation**
- [ ] Deploy emergency model patch if exploit identified
- [ ] Notify affected customers (transparent but careful)
- [ ] File SAR (Suspicious Activity Report) with FinCEN
- [ ] Draft press statement (if public disclosure required)

**Post-mortem (Week 1):**
- [ ] Root cause analysis (5 Whys)
- [ ] Action items with owners and deadlines
- [ ] Update playbooks based on lessons learned

**Communication:**
- **Customers:** "We detected and stopped fraudulent activity. Your account is secure."
- **Board:** Incident summary, financial impact, remediation plan
- **Media:** Only if legally required; otherwise no comment

---

## 🔹 For Risk/Ops Interviewers

### Q13.4: How do you measure the operational efficiency of the fraud team?

**A:**

**Operational metrics:**

| Metric | Formula | Target | Why It Matters |
|--------|---------|--------|----------------|
| **Review throughput** | Cases reviewed per analyst per day | ≥ 150 | Capacity planning |
| **Decision accuracy** | % of analyst decisions upheld on appeal | ≥ 95% | Quality assurance |
| **Time to decision** | Avg minutes from alert to resolution | ≤ 30 min | Customer experience |
| **Escalation rate** | % of cases escalated to senior analysts | ≤ 10% | Training effectiveness |
| **False positive rate** | % of blocked transactions later confirmed legitimate | ≤ 15% | Model calibration |
| **Cost per review** | Total ops cost / # cases reviewed | ≤ $8 | Efficiency |

**Dashboard example:**
```
Fraud Operations Dashboard (Last 7 Days)
─────────────────────────────────────────
Cases Reviewed:        12,450
Avg Time to Decision:  22 minutes
Decision Accuracy:     96.3%
Escalation Rate:       7.2%
Cost per Review:       $7.45

Top Alert Types:
1. Device reuse (>20 accounts): 3,200 cases
2. Velocity spike (>5x normal): 2,800 cases
3. Geographic anomaly: 1,900 cases
```

**Continuous improvement:**
- Weekly calibration sessions (review borderline cases)
- Monthly training on new fraud patterns
- Quarterly tooling upgrades (better UI, automation)

---

# 14. RISK & COMPLIANCE QUESTIONS

## 🔹 For Risk/Compliance Interviewers

### Q14.1: How do you ensure compliance with ECOA, FCRA, and fair lending laws?

**A:**

**Compliance framework:**

**1. Pre-deployment review:**
- Legal review of all features (no proxies for protected classes)
- Disparate impact testing (80% rule)
- Documentation of business necessity for each feature

**2. Ongoing monitoring:**
```python
# Monthly fairness audit
def fairness_audit():
    for protected_class in ['race', 'gender', 'age', 'zip_income']:
        di = calculate_disparate_impact(model, protected_class)
        eod = calculate_equal_opportunity_diff(model, protected_class)
        
        if di < 0.8 or abs(eod) > 0.1:
            trigger_investigation()
            file_compliance_report()
```

**3. Adverse action management:**
- Automated notice generation (FCRA §615(a))
- Specific reasons disclosed (top 4 SHAP features)
- Dispute process (60-day investigation window)

**4. Model risk management (SR 11-7):**
- Independent validation team (separate from developers)
- Annual model re-certification
- Change management (all updates documented and approved)

**5. Audit trail:**
- All predictions logged with features, scores, reasons
- Retained for 25 months (statute of limitations)
- Queryable for regulatory exams

---

### Q14.2: What is your model governance process?

**A:**

**Model Governance Board (MGB):**

**Composition:**
- Chair: Chief Risk Officer
- Members: VP Data Science, VP Engineering, General Counsel, Chief Compliance Officer, Head of Model Validation

**Meeting cadence:**
- **Quarterly:** Review all production models
- **Ad-hoc:** Emergency meetings for incidents or major changes

**Agenda:**
1. Model performance review (metrics vs expectations)
2. Fairness and compliance audit results
3. Incident review (if any)
4. Proposed changes (new features, retraining, threshold adjustments)
5. Retirement decisions (models to decommission)

**Decision rights:**
- **Approval:** Majority vote; CRO has veto on risk grounds
- **Escalation:** CEO decides if board deadlocked

**Documentation:**
- Model inventory (all models in production)
- Model cards (one-pager per model)
- Meeting minutes and decisions
- Issue tracker (open risks and mitigations)

---

### Q14.3: How do you prepare for an OCC or FDIC examination?

**A:**

**Exam preparation checklist:**

**1. Documentation (ready in 48 hours):**
- [ ] Model inventory with descriptions
- [ ] Model cards for all production models
- [ ] Validation reports (initial and annual)
- [ ] Performance monitoring dashboards
- [ ] Fairness audit results (last 12 months)
- [ ] Incident logs and post-mortems
- [ ] Change management records

**2. Access provisioning:**
- [ ] Read-only access to model code repository
- [ ] Query access to prediction logs
- [ ] Demo environment for model walkthrough

**3. Personnel:**
- [ ] Designated exam coordinator (usually Model Risk Manager)
- [ ] Subject matter experts on call (DS, Eng, Compliance)
- [ ] Legal counsel for privileged communications

**4. Common examiner requests:**
- "Show me how you prevent bias." → Fairness audit reports
- "Prove the model works." → Backtesting and A/B test results
- "What if the model fails?" → Disaster recovery plan
- "How do you handle customer disputes?" → Adverse action process

**Pro tip:** Examiners appreciate transparency. Proactively disclose known limitations and mitigation plans rather than waiting for them to discover issues.

---

### Q14.4: What is your policy on using alternative data (social media, utility payments, etc.)?

**A:**

**Alternative data policy:**

**Permitted (with approval):**
- ✅ Utility payment history (with customer consent)
- ✅ Rental payment history
- ✅ Cash flow data (bank transaction categorization)
- ✅ Education/employment verification (via trusted providers)

**Prohibited:**
- ❌ Social media data (privacy concerns, bias risk)
- ❌ Purchased contact lists (no customer relationship)
- ❌ Medical/health data (HIPAA violations)
- ❌ Data from data brokers without explicit consent

**Approval process:**
1. **Legal review:** Compliance with FCRA, state privacy laws (CCPA, VCDPA)
2. **Fairness testing:** Disparate impact analysis before deployment
3. **Customer consent:** Clear disclosure and opt-in (where required)
4. **MGB approval:** Model Governance Board sign-off

**Example:** We tested social media sentiment as a feature. Legal flagged privacy concerns; fairness testing showed disparate impact on older demographics. **Decision:** Do not deploy.

---

# 15. SCENARIO-BASED QUESTIONS

## 🔹 For All Audiences

### Q15.1: It's Black Friday. Transaction volume is 10x normal. Your model latency spikes to 500ms. What do you do?

**A:**

**Immediate actions (first 15 minutes):**
1. **Enable circuit breaker:** Switch to fallback mode (rule-based scoring)
2. **Auto-scale:** Kubernetes HPA should trigger; verify pods are scaling
3. **Shed load:** Prioritize high-value transactions (> $1K); queue low-value
4. **Alert:** Page on-call SRE and ML engineer

**Root cause investigation:**
- Check feature store (Redis) for hot keys or memory pressure
- Check model server CPU/memory utilization
- Check for upstream data pipeline bottlenecks

**Likely causes:**
- **Feature store overload:** Too many concurrent lookups
- **Model pod resource exhaustion:** CPU throttling
- **Network congestion:** Inter-service communication delays

**Fix options:**
1. **Short-term:** Increase Redis cluster size, add model pods
2. **Medium-term:** Cache frequently-accessed features (account-level aggregates)
3. **Long-term:** Pre-compute features for high-volume accounts (push, not pull)

**Post-mortem:**
- Why didn't load testing catch this? (Test at 5x, not 2x normal)
- Should we have better auto-scaling policies?
- Do we need a lighter-weight model for peak periods?

---

### Q15.2: A journalist claims your model discriminates against minority borrowers. How do you respond?

**A:**

**Crisis communication playbook:**

**Hour 0-4: Internal assessment**
- [ ] Verify the claim: Run fairness audit on disputed demographic
- [ ] Gather facts: What is the actual disparate impact?
- [ ] Legal review: Assess liability and disclosure obligations
- [ ] Prepare holding statement

**Hour 4-24: External response**
- [ ] Issue public statement: "We take these allegations seriously. We are conducting a thorough review and will share findings."
- [ ] Proactive outreach: Contact regulators before they contact you
- [ ] Customer communication: Email to affected segment (if confirmed)

**If claim is FALSE:**
- Publish fairness audit results (transparency report)
- Invite third-party audit (academic researchers, civil rights orgs)
- Consider legal action for defamation (if malicious)

**If claim is TRUE:**
- Acknowledge and apologize
- Announce remediation plan (model retraining, threshold adjustment)
- Commit to ongoing monitoring and public reporting
- Set up customer restitution fund (if harm occurred)

**Long-term:**
- Establish external advisory board (civil rights, consumer advocates)
- Publish annual fairness report
- Increase diversity in ML hiring

**Example:** Apple/Card Goldman Sachs (2019) faced similar allegations. Response: "We don't use gender in our algorithm." Later: Enhanced transparency. Lesson: Proactive fairness monitoring prevents crises.

---

### Q15.3: You discover label leakage: 30% of your "fraud" labels were based on model predictions, not confirmed fraud. What do you do?

**A:**

**This is a CRITICAL bug. Immediate actions:**

**Step 1: Stop the bleeding**
- [ ] Halt all model training pipelines
- [ ] Freeze current model in production (don't retrain until fixed)
- [ ] Notify leadership (VP DS, CRO, CEO)

**Step 2: Assess impact**
- [ ] Quantify leakage: Which features leaked? How much?
- [ ] Retrospective analysis: How long has this existed?
- [ ] Re-evaluate historical model performance (was it illusory?)

**Step 3: Fix the pipeline**
- [ ] Correct label definition: Only use confirmed fraud (chargebacks, investigations)
- [ ] Add data quality checks: Flag if labels correlate too highly with predictions
- [ ] Implement label provenance tracking (source of each label)

**Step 4: Retrain and validate**
- [ ] Retrain model with clean labels
- [ ] Expect performance drop (PR-AUC 0.71 → 0.60 is likely)
- [ ] Re-run all fairness and compliance tests

**Step 5: Communicate**
- [ ] Internal: Post-mortem with blameless culture
- [ ] External: If model impacted customers, consider disclosure
- [ ] Regulatory: If fair lending impacted, proactively notify

**Lesson:** Labels are ground truth. If they're corrupted, everything built on them is suspect. Invest in label quality infrastructure (human review workflows, audit trails).

---

### Q15.4: A competitor launches "zero-friction" onboarding with no verification. Your conversion rate drops 15%. How do you respond?

**A:**

**Strategic response options:**

**Option 1: Match them (NOT recommended)**
- Remove verification for low-risk segments
- **Risk:** Fraud losses could spike 3-5x; race to the bottom

**Option 2: Differentiate on security (RECOMMENDED)**
- Marketing campaign: "Safe banking is smart banking"
- Highlight fraud protection as a feature, not a bug
- Offer fraud insurance guarantee (we cover unauthorized transactions)

**Option 3: Hybrid approach**
- Zero-friction for existing customers (trust established)
- Streamlined verification for new customers (2-step vs 5-step)
- Premium tier: Paid subscription for expedited onboarding

**Option 4: Call their bluff**
- Wait for their fraud losses to emerge (they will)
- Meanwhile, optimize our friction (reduce from 5% to 3%)
- Prepare case studies when they have public fraud incidents

**Recommended playbook:**
1. **Short-term:** Reduce friction by 20% (remove one verification step)
2. **Medium-term:** Launch marketing emphasizing security
3. **Long-term:** Monitor competitor; expect them to tighten after fraud spike

**Data point:** Zelle faced fraud issues after rapid growth with minimal verification. They later added more controls. **Lesson:** Security can't be sacrificed for growth indefinitely.

---

### Q15.5: You're launching in a new market (e.g., BNPL — Buy Now Pay Later) with no historical fraud data. How do you build a model?

**A:**

**Cold start strategy:**

**Phase 1: Rules + Transfer Learning (Month 1-2)**
- **Rules-based:** Industry benchmarks (block transactions > $2K for new accounts)
- **Transfer learning:** Use embeddings from core banking product (similar behaviors)
- **Conservative thresholds:** High friction initially; relax as data accumulates

**Phase 2: Semi-supervised Learning (Month 3-4)**
- **Label propagation:** Use graph structure to infer fraud (accounts connected to known fraudsters)
- **Active learning:** Manually review 500 cases/week; prioritize uncertain predictions
- **Synthetic fraud:** Simulate fraud scenarios based on industry patterns

**Phase 3: Supervised ML (Month 5-6)**
- By now, have 1K+ labeled fraud cases
- Train LightGBM with features from Phases 1-2
- A/B test against rules baseline

**Key features for BNPL:**
- **Merchant risk:** Some merchants have 10x higher fraud rates
- **Device fingerprinting:** New device + high amount = red flag
- **Velocity:** Multiple BNPL applications in 24 hours
- **Graph:** Shared devices/addresses across applications

**Risk mitigation:**
- Start with low credit limits ($500)
- Require bank account verification (Plaid)
- Partner with credit bureaus for thin-file scoring

**Timeline to maturity:** 6 months to reach 60% fraud detection (vs 70% in core product).

---

### Q15.6: Your best ML engineer quits and joins a startup. They have access to all your model code and data. What do you do?

**A:**

**Offboarding security checklist:**

**Immediate (Day 0):**
- [ ] Revoke all access (code repos, data warehouses, cloud consoles)
- [ ] Change shared credentials they knew
- [ ] Retrieve company devices (laptop, badge)

**Short-term (Week 1):**
- [ ] Audit access logs: What did they download in last 30 days?
- [ ] Review NDAs and restrictive covenants (non-compete, non-solicit)
- [ ] Legal briefing: What can/can't they do at new company?

**Medium-term (Month 1-3):**
- [ ] Knowledge transfer: Ensure documentation is complete
- [ ] Bus factor mitigation: Cross-train other engineers
- [ ] Model security: Consider rotating model architectures (if they know internals)

**Long-term:**
- [ ] Exit interview: Why are they leaving? (Address systemic issues)
- [ ] Retention program: Competitive comp, equity, career growth
- [ ] Alumni network: Maintain positive relationship (boomerang hires)

**Legal protections:**
- **NDA:** Prohibits sharing confidential information
- **Invention assignment:** Company owns all work product
- **Non-compete:** Enforceable in some states (CA limits enforceability)

**Reality check:** Most employees leave amicably. Assume good faith but verify. If they join a direct competitor and you suspect IP theft, consult legal about litigation options.

---

# 16. QUICK REFERENCE: MODEL COMPARISON MATRIX

## 🔹 Decision Guide: Which Model to Use?

| Scenario | Recommended Model | Why | Latency | Accuracy (PR-AUC) |
|----------|------------------|-----|---------|-------------------|
| **Tabular transaction data** | LightGBM | Best speed/accuracy trade-off | 15ms | 0.71 |
| **Identity ring detection** | GraphSAGE + LightGBM | Captures graph structure | 45ms | 0.78 |
| **Behavioral sequence patterns** | Transformer | Models temporal dependencies | 120ms | 0.75 |
| **Cold start (no labels)** | Isolation Forest | Unsupervised anomaly detection | 10ms | 0.55 |
| **Real-time scoring (<20ms)** | LightGBM (pruned) | Optimized for latency | 12ms | 0.68 |
| **Explainability required** | Logistic Regression + SHAP | Fully interpretable | 5ms | 0.62 |
| **Extreme class imbalance** | LightGBM + Focal Loss | Handles 1000:1 imbalance | 15ms | 0.70 |
| **Multi-task (fraud + AML)** | Multi-task Neural Net | Shared representations | 80ms | 0.73 |

---

## 🔹 Metric Selection Guide

| Goal | Primary Metric | Secondary Metric | Guardrail |
|------|---------------|------------------|-----------|
| **Maximize fraud caught** | Recall @ K | Precision @ K | FP Rate < 2% |
| **Minimize false positives** | Precision | Specificity | Recall > 60% |
| **Optimize business cost** | Cost per Transaction | ROI | NPS impact |
| **Detect novel fraud** | Anomaly Score Distribution | Unsupervised Recall | Human review rate |
| **Ensure fairness** | Disparate Impact Ratio | Equal Opportunity Diff | DI ≥ 0.8 |
| **Monitor drift** | PSI | KS Statistic | PSI < 0.2 |

---

## 🔹 Threshold Selection Guide

| Business Objective | Recommended Threshold | Expected Recall | Expected Precision |
|--------------------|----------------------|-----------------|-------------------|
| **Aggressive fraud prevention** | 0.15 | 85% | 8% |
| **Balanced (default)** | 0.23 | 68% | 12% |
| **Customer experience focus** | 0.45 | 45% | 22% |
| **Manual review queue (triage)** | 0.30 | 60% | 15% |
| **Auto-block (no review)** | 0.70 | 25% | 65% |

**Note:** Thresholds should be recalibrated quarterly based on business cost optimization.

---

## 🔹 Glossary of Terms

| Term | Definition |
|------|------------|
| **PR-AUC** | Area under Precision-Recall curve; preferred for imbalanced data |
| **PSI** | Population Stability Index; measures distribution shift |
| **Node2Vec** | Graph embedding algorithm; random walks + Skip-gram |
| **Focal Loss** | Modified cross-entropy that down-weights easy negatives |
| **Disparate Impact** | Ratio of favorable outcomes between protected/unprotected groups |
| **Training-Serving Skew** | Difference between training and production data distributions |
| **Watermarking** | Mechanism to handle late-arriving data in streaming |
| **Delta Lake** | Storage layer providing ACID transactions on data lakes |
| **Feature Store** | Centralized repository for consistent feature computation |
| **Canary Deployment** | Gradual rollout to subset of users before full deployment |

---

## 📞 Contact & Resources

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Maintained By:** Data Science & Risk Teams  

**Additional Resources:**
- Model Cards Repository: `/confluence/model-cards`
- Fraud Playbook: `/confluence/fraud-response`
- API Documentation: `/swagger/fraud-detection`
- On-Call Rotation: `/pagerduty/fraud-service`

---

*This document is confidential and intended for internal interview preparation only. Do not distribute externally.*
