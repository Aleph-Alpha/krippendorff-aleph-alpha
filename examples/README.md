# Data Types and Examples

This section explains various types of data that can be used for measuring inter-annotator agreement using Krippendorff's alpha. The examples showcase the types of tasks and how agreement and disagreement is calculated.

The example datasets in this directory have 3 or more annotators and the examples below are for references only.

## 1. Nominal Data (Categorical, No Order)

Labels are simply categories, and there is no inherent ranking between them. Disagreement is binary: either the labels match (agreement) or they don't (disagreement).

### ✅ Example: Named Entity Recognition (NER)
| Text                | Annotator 1 | Annotator 2 |
|---------------------|-------------|-------------|
| "Apple"             | ORG         | PERSON      |
| "New York"          | LOCATION    | LOCATION    |
| "car"               | O (None)    | O (None)    |

- "Apple" is labeled as `ORG` by Annotator 1 and `PERSON` by Annotator 2 → **Disagreement** (1).
- "New York" is labeled the same (`LOCATION`) → **Agreement** (0).

### 🔹 Used for:
- Named Entity Recognition (NER)
- Part-of-Speech (POS) tagging
- Document Classification (e.g., Spam vs. Not Spam)

---

## 2. Ordinal Data (Ordered Categories, Unequal Gaps)

Labels have a ranking, but the differences between the ranks are not necessarily equal.

### ✅ Example: Sentiment Classification
| Text                          | Annotator 1    | Annotator 2    |
|-------------------------------|----------------|----------------|
| "This movie is ok."            | Neutral        | Positive       |
| "Absolutely loved it!"         | Very Positive  | Positive       |
| "Horrible experience."         | Very Negative  | Negative       |

- "Neutral" (rank 2) vs. "Positive" (rank 3) → **Disagreement**, but not as severe as "Very Negative" (rank 0) vs. "Positive" (rank 3).
- The squared difference in rank is used to compute disagreement.

### 🔹 Used for:
- Sentiment Analysis (Very Negative → Very Positive)
- Star Ratings (1⭐ to 5⭐)
- Readability Levels (Beginner → Advanced)

---

## 3. Interval Data (Numeric, Equal Gaps, No Absolute Zero)

Differences between values are meaningful, but there is no true zero.

### ✅ Example: Readability Scores
| Text                    | Annotator 1 | Annotator 2 |
|-------------------------|-------------|-------------|
| "The cat sat."           | 60.0        | 55.0        |
| "Complex passage"        | 40.0        | 50.0        |
| "Scientific paper"       | 20.0        | 22.0        |

- The difference of 5 points (60 → 55) is as important as a difference of 5 points (40 → 35).
- Squared differences (v1 - v2)² are used to compute disagreement.

### 🔹 Used for:
- Readability Scores
- Emotion Intensity (Scale 0–100)
- Machine Translation Quality Scores

---

## 4. Ratio Data (Numeric, Equal Gaps, Absolute Zero Exists)

Like interval data, but with a true zero value that means "none" of the property exists.

### ✅ Example: Offensive Word Count
| Text               | Annotator 1 | Annotator 2 |
|--------------------|-------------|-------------|
| "Nice day."        | 0           | 0           |
| "You idiot!"       | 2           | 3           |
| "What the hell!"   | 1           | 2           |

- If Annotator 1 says 2 offensive words and Annotator 2 says 3, the disagreement is computed as:  
  `(∣v1−v2∣ / (v1+v2))²`

### 🔹 Used for:
- Offensive Language Detection
- Number of Grammar Errors in a Sentence
- Word Count in a Specific Category (e.g., adjectives in a sentence)

---

## Summary Table

| Data Type | Example Task           | Example Labels/Values             | Agreement Calculation        |
|-----------|------------------------|-----------------------------------|------------------------------|
| Nominal   | NER, POS, Classification| PERSON, ORG, O                    | Binary (0 or 1)              |
| Ordinal   | Sentiment Analysis      | Very Negative (0) → Very Positive (4) | Squared rank differences     |
| Interval  | Readability, MT Quality| Scores (e.g., 0-100)              | Squared numeric differences  |
| Ratio     | Offensive Word Count    | 0, 1, 2, 3…                       | Normalized squared differences |
