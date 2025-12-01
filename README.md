# Social Media Post Recommender System

A hybrid machine learning system that helps content creators discover high-performing social media posts using semantic search, engagement prediction, and LLM-powered insights.

## Project Overview

This recommender system combines FAISS vector search with SVM classification to retrieve and rank social media posts based on both semantic similarity and predicted engagement potential. The system analyzes 12,000 social media posts to identify patterns and provide actionable recommendations for content creators.

**Team:** Andres Castellanos, Mora Dominguez, Sigurdur Sigurdsson  
**Course:** Machine Learning Final Project  
**Instructor:** Dr. Ananda M. Mondal  
**Dataset:** Kaggle Social Media Engagement Dataset (12,000 posts)

## System Architecture

The system implements a 4-stage pipeline:

1. **Vector Database (FAISS)**: Semantic embeddings using Sentence Transformers (all-MiniLM-L6-v2)
2. **Semantic Retrieval**: Top-K search with engagement filtering (75th percentile threshold: 0.163)
3. **Hybrid Ranking**: Combines FAISS similarity scores with candidate-level SVM engagement predictions
4. **LLM Explanation**: Pattern analysis and actionable insights using Claude/GPT-4

### Key Technical Decisions

- **Data-driven thresholds** (percentile-based):
  - High engagement: ≥0.41 (90th percentile)
  - Medium engagement: 0.08-0.41 (50th-90th percentile)
  - Low engagement: <0.08 (below median)
  - FAISS filter: >0.163 (75th percentile - top 25%)

- **Balanced class distribution**: 50% Low / 40% Medium / 10% High
  - Prevents severe class imbalance (original arbitrary thresholds created 93%/4.6%/1.95% split)
  - Ensures sufficient training examples for SVM classifier

- **Candidate-level scoring**: Each of the 20 retrieved posts receives its own classifier score
  - Not query-level scoring (which would give all posts the same score)

## Project Structure
```
FinalProject/
├── README.md
├── artifacts/
│   ├── stage2_results/
│   │   └── sample_stage2_results.pkl
│   └── stage3_results/               # currently empty
├── models/
│   ├── post_embeddings.npy
│   ├── posts_faiss.index
│   └── posts_metadata.pkl
├── notebooks/
│   ├── 00_social_media_preprocessing_updated.ipynb
│   ├── 01_build_vector_database.ipynb
│   └── 02_semantic_retrieval_updated.ipynb
├── processed_data/
│   ├── social_media_processed.csv
│   ├── test_data.csv
│   ├── thresholds.txt
│   ├── train_data.csv
│   └── val_data.csv
├── src/                              # reserved for scripts (empty)
└── .gitignore
```

## Data Preprocessing

### Engagement Distribution Analysis

The dataset exhibits a highly skewed power-law distribution:
- **Median**: 0.0806 (50% of posts)
- **75th percentile**: 0.1631 (FAISS filter threshold)
- **90th percentile**: 0.4119 (High label threshold)
- **96%** of posts have engagement ≤ 1.0
- **4.06%** are outliers (>1.0, representing viral posts - kept in dataset)

### Preprocessing Steps

1. **Data cleaning**: Handle missing values, remove duplicates, validate engagement metrics
2. **Text combination**: Create `combined_text` field from `text_content` + `hashtags`
3. **Engagement labeling**: Apply percentile-based thresholds to create balanced classes
4. **Stratified splitting**: 60/20/20 split maintaining class distribution across sets
5. **Outlier handling**: Keep all 487 viral posts (engagement > 1.0) - they represent real phenomena

## Model Components

### 1. Sentence Transformer (Embeddings)
- **Model**: all-MiniLM-L6-v2
- **Dimension**: 384
- **Normalization**: L2 (enables cosine similarity via inner product)

### 2. FAISS Vector Database
- **Index type**: IndexFlatIP (Inner Product)
- **Size**: 12,000 vectors × 384 dimensions
- **Search time**: <50ms for top-50 candidates

### 3. SVM Classifier
- **Features**: TF-IDF (5,000 max features, unigrams + bigrams)
- **Kernel**: Linear SVM with probability estimates
- **Classes**: High / Medium / Low engagement
- **Target accuracy**: >65% (vs 33% random baseline)

### 4. Hybrid Ranking Mechanism
```python
ranking_score = (0.5 × faiss_score) + (0.5 × classifier_score)
```
- **faiss_score**: Cosine similarity between query and post
- **classifier_score**: P(High | post_text) for each specific candidate post
- Both scores are post-specific and computed independently

## Getting Started

### Prerequisites
```bash
# Python 3.8+
pip install pandas numpy scikit-learn
pip install sentence-transformers faiss-cpu
pip install anthropic  # For LLM integration
pip install kagglehub  # For dataset download
```

### Data Acquisition
```python
import kagglehub
# Download dataset
path = kagglehub.dataset_download("subashmaster0411/social-media-engagement-dataset")
```

### Running the Pipeline

1. **Data preprocessing** (completed):
   - Processed dataset saved to `processed_data/`
   - Balanced class distribution achieved
   - Stratified train/val/test splits created

2. **Build FAISS index** (next step):
   - Generate embeddings for all 12,000 posts
   - Create and save FAISS index
   - Store metadata dictionary

3. **Train SVM classifier**:
   - Fit TF-IDF vectorizer on training set
   - Train linear SVM with balanced class weights
   - Evaluate on validation set

4. **Implement hybrid ranking**:
   - Integrate FAISS retrieval with SVM scoring
   - Tune fusion weights (default: 0.5/0.5)

5. **Add LLM explanation**:
   - Format top-5 posts into structured prompt
   - Generate pattern analysis and recommendations

## Evaluation Metrics

### Retrieval Quality
- **Precision@5**: Target >0.80
- **NDCG@10**: Target >0.75
- **Recall@10**: Target >0.70

### Classification Performance
- **Accuracy**: Target >65%
- **F1-scores** per class (High/Medium/Low)
- Confusion matrix analysis

### Ranking Improvement
- **A/B test**: FAISS-only vs Hybrid ranking
- **Expected improvement**: 15-20% in average engagement of top-5 results

### Human Evaluation
- Relevance, quality, and actionability ratings (1-5 scale)
- Target scores: >4.0 relevance, >4.2 quality, >3.8 actionability

## Key Updates from Original Plan

| Component | Original | Updated | Rationale |
|-----------|----------|---------|-----------|
| High threshold | ≥0.7 | ≥0.41 (90th pct) | Avoids severe class imbalance |
| Medium threshold | 0.4-0.7 | 0.08-0.41 (50th-90th) | Creates trainable classes |
| FAISS filter | >0.6 | >0.163 (75th pct) | More candidates (25% vs 5%) |
| Outlier handling | Not specified | Keep all (4.06%) | Viral posts are valid data |
| Scoring approach | Unclear | Candidate-level | Each post scored individually |
| Terminology | Cross-encoder | Hybrid ranking | Accurate technical description |

## Limitations

- **Dataset**: Snapshot in time, trends evolve
- **Cold start**: New hashtags/topics not in training data
- **Static model**: No online learning post-deployment
- **Thresholds**: Dataset-specific, may not generalize
- **Language**: English-only in current implementation

## Future Enhancements

1. **Engagement prediction** for user's draft content
2. **Temporal trend analysis** tracking rising/falling topics
3. **Multi-modal search** with image/video similarity
4. **Personalization** via user profile learning
5. **Active learning** with continuous model improvement
6. **Dynamic thresholds** adapting to new data distributions

## License

Academic project for educational purposes.

## Acknowledgments

- Dataset: Kaggle Social Media Engagement Dataset (subashmaster0411)
- Instructor: Dr. Ananda M. Mondal
- Sentence Transformers: all-MiniLM-L6-v2
