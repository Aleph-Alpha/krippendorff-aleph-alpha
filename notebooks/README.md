# Krippendorff’s Alpha Usability Testing Notebook 📝📊

## Overview

This Jupyter Notebook provides an interactive environment for testing Krippendorff’s Alpha computation. It allows users to:

-  Load and preprocess annotation datasets.
-  Compute the reliability matrix from annotation data. 
-  Calculate Krippendorff’s Alpha for different metrics (nominal, ordinal, interval, ratio).  
-  Inspect per-category agreement scores.

This is useful for validating inter-annotator agreement before applying annotations to machine learning or linguistic analysis tasks.

------
## Requirements

Before running the notebook, ensure you have all dependencies installed. If you’re using uv, sync dependencies with:

```bash
uv sync
```

----

## How to Use

### 1️⃣ Open the Notebook

Launch Jupyter Notebook in the project root:

```bash
jupyter notebook notebooks/krippendorff_alpha_demo.ipynb

```

### 2️⃣ Import the Necessary Functions

The notebook will automatically import:

```bash
from src.krippendorff_alpha.config import compute_alpha
```

### 3️⃣ Load Your Data

You can either provide a dataset file path or manually create a pandas.DataFrame.
Refer to the examples directory to get more information on supported examples.

Examples:
```bash
results = compute_alpha(path="examples/nominal_categorical_noOrder_sample.json"
```

or in-memory dataframe:

```bash

import pandas as pd

df = pd.DataFrame({
    "word": ["apple", "banana", "cherry"],
    "annotator1": ["red", "yellow", "red"],
    "annotator2": ["red", "yellow", "red"],
    "annotator3": ["green", "yellow", "red"],
})

results = compute_alpha(df=df, text_col="word", annotator_cols=["annotator1", "annotator2", "annotator3"])
```

### 4️⃣ Analyze the Results

The notebook will print results like:

    {
        "alpha": 0.85,
        "observed_disagreement": 0.12,
        "expected_disagreement": 0.80,
        "per_category_scores": {
            "red": {"observed_disagreement": 0.1, "expected_disagreement": 0.5},
            "yellow": {"observed_disagreement": 0.0, "expected_disagreement": 0.1},
        }
    }

