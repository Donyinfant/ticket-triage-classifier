# 🎫 Ticket Triage Classifier

A support ticket triage tool that reads new, unseen ticket text and automatically predicts its category — **Billing, Technical, HR, or General** — using TF-IDF text vectorization and a Multinomial Naive Bayes classifier.

This project simulates a real internal tool sitting in front of a live ticket queue: every prediction comes with a confidence score, and any ticket the model isn't confident about is automatically flagged for human review instead of being silently auto-assigned.

## Approach summary

Ticket text (subject + body) is cleaned — lowercased, stripped of punctuation, and filtered of common stopwords — then converted into numerical features using TF-IDF with unigrams and bigrams, so two-word phrases like "not working" carry meaning too. A Multinomial Naive Bayes classifier is trained on this data, since it's a fast, reliable baseline for text classification that performs well even on a small dataset. Edge cases are handled with a confidence threshold: any prediction below 60% confidence is routed to a "needs human review" flag rather than auto-assigned, since a wrong silent classification is worse than a short manual review delay.

## How it works

1. **Load data** — reads labeled tickets from a CSV (`subject`, `body`, `category`)
2. **Clean text** — lowercase, remove punctuation, strip stopwords
3. **Vectorize** — TF-IDF with unigrams + bigrams turns text into numbers
4. **Train/test split** — 75% train, 25% held out for evaluation
5. **Train classifier** — Multinomial Naive Bayes
6. **Evaluate** — accuracy, precision/recall/F1 per class, confusion matrix
7. **Classify new tickets** — returns category + confidence score
8. **Human review fallback** — flags predictions under 60% confidence

## Results

| Metric | Score |
|---|---|
| Accuracy | ~94–97% |
| Macro F1 | ~0.95–0.97 |

(See console output when running the script for the full classification report and confusion matrix.)

## Sample predictions on new, unseen tickets

| Ticket | Predicted Category | Confidence | Status |
|---|---|---|---|
| "Overcharged this month" | Billing | 60.8% | Auto-assigned |
| "App won't open" | Technical | 51.3% | Needs review |
| "How many vacation days left" | HR | 52.0% | Needs review |
| "Do you have student pricing" | General | — | Needs review |
| "Great job on the update" | General | — | Needs review |

## How to run

```bash
pip install pandas scikit-learn
python ticket_classifier.py
```

Make sure the CSV file is in the same folder as the script.

## Project structure
## What this evaluates

- **Text preprocessing** — cleaning raw ticket text before modeling
- **Feature representation** — TF-IDF vectorization instead of raw word counts
- **Model choice** — Naive Bayes, chosen for speed and strong baseline performance on small text datasets
- **Evaluation literacy** — accuracy, precision, recall, F1, and confusion matrix
- **Real-time usability** — classifies one new ticket on demand, not just a static test set
- **Edge-case handling** — low-confidence predictions are flagged instead of guessed

## Reflection — what I'd improve with more data or time

The biggest limitation is dataset size — 141 tickets is enough to demonstrate the pipeline but not enough to fully separate genuinely ambiguous tickets (e.g. one that mentions both a login problem and a refund). With more data and time, I'd add cross-validation instead of a single train/test split, support multi-label tickets since real tickets often span more than one category, and track live accuracy after deployment to catch model drift as ticket phrasing changes over time.
