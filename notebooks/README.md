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

###  Open the Notebook

Launch Jupyter Notebook in the project root:

```bash
jupyter notebook notebooks/krippendorff_alpha_demo.ipynb

```
----
### Setting Up Jupyter for This Project

If you encounter issues running the notebooks (e.g., missing modules), follow these steps to set up Jupyter with the correct Python environment.

### Step 1: Install Jupyter in Your Project Environment

Run the following in your project's terminal:

```bash
uv pip install jupyter ipykernel

```

### Step 2: Add Your Project’s Python as a Jupyter Kernel

Run:
```bash
python -m ipykernel install --user --name=krippendorff-aleph-alpha --display-name "Python (krippendorff)"

```

### Step 3: Restart Jupyter and Select the Correct Kernel

1. Restart Jupyter:

```bash

jupyter lab  # Or jupyter notebook

```
2. Open your notebook (annotation_evaluation.ipynb).

3. Go to Kernel → Change Kernel.

4. Select "Python (krippendorff)" (the one you just installed).


### Step 4: Verify the Fix

In your Jupyter notebook, run:

```bash

import sys
print(sys.executable)

```