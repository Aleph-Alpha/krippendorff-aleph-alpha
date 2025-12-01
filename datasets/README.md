# Data Types and Datasets

This directory contains example datasets for measuring inter-annotator agreement using Krippendorff's alpha. All datasets are in TSV (tab-separated values) format with standardized column names.

## Dataset Format

All datasets follow this format:
- **First column**: `text` - Contains the text being annotated
- **Subsequent columns**: `annotator1`, `annotator2`, `annotator3`, etc. - Contain annotations from each annotator
- **Minimum requirements**: At least 3 annotator columns and 3 text units

## Available Datasets

### 1. Nominal Data (Categorical, No Order)
**File**: `nominal_categorical_noOrder_sample.tsv`

Labels are simply categories, and there is no inherent ranking between them. Disagreement is binary: either the labels match (agreement) or they don't (disagreement).

**Example**: Named Entity Recognition (NER)
- "Apple" labeled as `ORG` vs `PERSON` → **Disagreement** (1)
- "New York" labeled as `LOCATION` by both → **Agreement** (0)

**Used for**:
- Named Entity Recognition (NER)
- Part-of-Speech (POS) tagging
- Document Classification (e.g., Spam vs. Not Spam)

---

### 2. Ordinal Data (Ordered Categories)

#### High Agreement Example
**File**: `ordinal_orderedCategories_highAgreement_sample.tsv`

Shows ordinal data with high inter-annotator agreement.

**Ordinal Scale**: Poor < Fair < Good < Very Good < Excellent

#### Unequal Gaps Example
**File**: `ordinal_orderedCategories_unequalGaps_sample.tsv`

Shows ordinal data with varying levels of disagreement, demonstrating how rank differences affect disagreement calculation.

**Ordinal Scale**: Very Negative < Negative < Neutral < Positive < Very Positive

**Key Points**:
- "Neutral" (rank 2) vs. "Positive" (rank 3) → Smaller disagreement than "Very Negative" (rank 0) vs. "Positive" (rank 3)
- The squared difference in rank is used to compute disagreement

**Used for**:
- Sentiment Analysis (Very Negative → Very Positive)
- Star Ratings (1⭐ to 5⭐)
- Readability Levels (Beginner → Advanced)

---

### 3. Interval Data (Numeric, Equal Gaps, No Absolute Zero)
**File**: `interval_numeric_equalGaps_noAbsoluteZero.tsv`

Differences between values are meaningful, but there is no true zero point.

**Example**: Readability Scores
- The difference of 5 points (60 → 55) is as important as a difference of 5 points (40 → 35)
- Squared differences (v1 - v2)² are used to compute disagreement

**Used for**:
- Readability Scores
- Emotion Intensity (Scale 0–100)
- Machine Translation Quality Scores

**Note**: These scores don't have a true zero (0 doesn't mean "no readability"), making them interval data rather than ratio data.

---

### 4. Ratio Data (Numeric, Equal Gaps, Absolute Zero Exists)
**File**: `ratio_numeric_equalGaps_withAbsoluteZero.tsv`

Like interval data, but with a true zero value that means "none" of the property exists.

**Example**: Offensive Word Count
- Zero (0) means no offensive words
- If Annotator 1 says 2 offensive words and Annotator 2 says 3, the disagreement is computed as: `(v1−v2)² / (v1+v2)`

**Used for**:
- Offensive Language Detection
- Number of Grammar Errors in a Sentence
- Word Count in a Specific Category (e.g., adjectives in a sentence)

**Note**: Zero values are meaningful here (no offensive words), making this ratio data rather than interval data.

---

## Summary Table

| Data Type | Example Task           | Example Labels/Values             | Disagreement Calculation        |
|-----------|------------------------|-----------------------------------|------------------------------|
| Nominal   | NER, POS, Classification| PERSON, ORG, O                    | Binary (0 or 1)              |
| Ordinal   | Sentiment Analysis      | Very Negative → Very Positive     | Squared rank differences     |
| Interval  | Readability, MT Quality| Scores (e.g., 0-100)              | Squared numeric differences  |
| Ratio     | Offensive Word Count    | 0, 1, 2, 3…                       | Normalized squared differences |

---

## Loading Datasets

These TSV files can be loaded using pandas:

```python
import pandas as pd

# Load a dataset
df = pd.read_csv('datasets/nominal_categorical_noOrder_sample.tsv', sep='\t')

# The DataFrame will have columns: text, annotator1, annotator2, annotator3, ...
```

## Notes

- All datasets use tab-separated values (TSV) format
- Column names are standardized: `text`, `annotator1`, `annotator2`, etc.
- For ordinal data, ensure you specify the correct ordinal scale order when computing alpha

