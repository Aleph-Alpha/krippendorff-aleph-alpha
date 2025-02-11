# Krippendorff's Aleph Alpha

🚧 Project in Progress 🚧

This repository contains an implementation of Krippendorff’s Alpha, a statistical measure of inter-rater reliability designed for various data types (Nominal, Ordinal, Interval, and Ratio). The goal is to support multiple data formats (CSV, JSON, Pandas DataFrames), efficient computations, and dynamic updates for annotators.


Features (Planned & In Progress):

✔️ Support for multiple data types (Nominal, Ordinal, etc.)

✔️ Support for reliability matrix storage and dynamic update

✔️ Flexible data input formats (CSV, JSON, DataFrames)

✔️ Custom weights for different applications

✔️ Performance optimizations (vectorization, sparse matrices)

✔️ Structured outputs with optional visualizations
Installation


## Setup
This project uses UV for dependency management. 

Steps:

Install UV using [UV Install Guide](https://docs.astral.sh/uv/getting-started/installation/)

After installing UV, you can then follow the instruction below.

Run:

```bash
uv run
```


Set Python interpreter in IDE:

```bash
krippendorff-aleph-alpha/.venv/bin/python3.13
```

Done. You're set up!

---------------

## Project Structure

### Project Structure Overview

The project is structured to ensure **modularity, scalability, and ease of use**. Below is a breakdown of the key components:

- **`src/krippendorff_alpha/`**  
  Contains the core implementation of Krippendorff's Alpha, including modules for preprocessing, reliability computation, and metric calculation.

- **`config.py`**  
  Central configuration file for computing Krippendorff's Alpha. It handles data loading, preprocessing, reliability matrix updates, and alpha computation.

- **`reliability.py`**  
  Manages the reliability matrix, including loading, updating, and computing it.

- **`preprocessing.py`**  
  Handles data preprocessing for different input formats and annotation levels.

- **`metric.py`**  
  Implements the Krippendorff's Alpha calculation for different metrics (**nominal, ordinal, interval, ratio**).

- **`schema.py`**  
  Defines data schemas and mappings for consistent data handling.

- **`utils.py`**  
  Utility functions for common operations.

- **`examples/`**  
  Contains example datasets in various formats (**CSV, JSON, TSV**) to demonstrate the usage of Krippendorff's Alpha for different data types (**nominal, ordinal, interval, ratio**).  
  Each dataset is accompanied by a description in the `README.md` file within the directory.

-------
## Jupyter Notebook (Upcoming)

A Jupyter notebook will be added to demonstrate the usage of Krippendorff's Alpha on various datasets. The notebook will:
