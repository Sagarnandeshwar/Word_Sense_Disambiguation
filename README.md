# Word Sense Disambiguation (WSD)

This project implements and compares four different Word Sense Disambiguation (WSD) methods to determine the meaning of a word based on its context. The models are evaluated using standard benchmarks and synthetically generated data.

## Objective

- Disambiguate word meanings in context using multiple methods.
- Compare model performances based on accuracy.

## Dataset

- **SemEval 2013 Task 12** subset (Navigli and Jurgens, 2013)
- **Boot Dataset**: 10 sample sentences per lemma generated using the large T5 model
- **Lexical Resource**: WordNet v3.0

## Preprocessing

- Lemmatization
- Stop word removal
- Punctuation removal
- Tokenization
- POS tagging
- Feature vector construction (Bag-of-Words)

## Models Implemented

### 1. Most Frequent Sense (Baseline)
- Selects the first synset from WordNet for each lemma.

### 2. Lesk Algorithm
- Calculates overlap between context and dictionary definitions for each candidate sense.

### 3. Supervised SVM Classifier
- One SVM per lemma.
- Trained on dev + boot dataset.
- Boot data labeled using Model 1 or Model 2.

### 4. Bootstrapped Decision Tree (Modified Yarowsky)
- Multi-layer bootstrapping with confidence thresholds.
- Uses Decision Trees and expands seed set across three layers.

## Results

| Model       | Dev Set Accuracy | Test Set Accuracy |
|-------------|------------------|-------------------|
| Model 1     | 0.675            | 0.623             |
| Model 2     | 0.418 (with POS) | 0.362 (with POS)  |
| Model 3     | 0.33–0.339       | N/A               |
| Model 4     | 0.599–0.619      | N/A               |

*Model 4 improves accuracy through 3 bootstrap layers.*

## Sample Outputs

Sample predictions for each model are included in the report, showing:
- Target word
- Context sentence
- Correct definition
- Predicted definition

## Observations

- **Model 1** outperforms **Model 2**, suggesting most sentences use the most common sense.
- **POS tagging** has limited impact on Lesk’s accuracy.
- **Model 3** accuracy remains low due to weak boot labels.
- **Model 4** shows improved accuracy via iterative learning.

## Suggestions for Improvement

- Manually curated datasets could improve supervised models.
- Multi-feature ML models can be explored for efficiency.
- Automating bootstrap layers could enhance scalability.

## References

1. “SemEval-2013 Task 12: Multilingual Word Sense Disambiguation” – Roberto Navigli et al.
2. “An Adapted Lesk Algorithm for Word Sense Disambiguation Using WordNet” – Satanjeev Banerjee et al.
3. “Analysis of Semi-Supervised Learning with the Yarowsky Algorithm” – Gholam Reza Haffari et al.
4. “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation” – Yusong Wu et al.

---

> Developed by [Sagar Nandeshwar]
