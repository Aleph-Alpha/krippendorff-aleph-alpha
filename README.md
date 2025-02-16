# Krippendorff's Aleph Alpha


This repository contains an implementation of Krippendorff’s Alpha, a statistical measure of inter-rater reliability designed for various data types (Nominal, Ordinal, Interval, and Ratio). 
Features (Planned & In Progress):

✔️ Support for multiple data types (Nominal, Ordinal, etc.)

✔️ Custom weights for annotator impact on annotation assignments

✔️ Performance optimizations 

✔️ Fine-grained output for categorical values




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

- **`examples/`**  
  Contains example datasets in various formats (**CSV, JSON, TSV**) to demonstrate the usage of Krippendorff's Alpha for different data types (**nominal, ordinal, interval, ratio**).  
  Each dataset is accompanied by a description in the `README.md` file within the directory.

- **`notebooks/`**  
  Contains usability examples for the code.

-------
## Jupyter Notebook 

A Jupyter notebook will be added to demonstrate the usage of Krippendorff's Alpha on various datasets. The notebook will:
