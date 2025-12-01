# Krippendorff's Alpha

A Python implementation of Krippendorff's Alpha, a statistical measure of inter-rater reliability designed for various data types (Nominal, Ordinal, Interval, and Ratio). 


✔️ Support for multiple data types (Nominal, Ordinal, etc.)

✔️ Custom weights for annotator impact on annotation assignments

✔️ Performance optimizations 

✔️ Fine-grained output for categorical values

This project is inspired by the following open-source implementations of Krippendorff's Alpha listed below:

- [Fast Krippendorff](https://github.com/pln-fing-udelar/fast-krippendorff)
- [Krippendorff Alpha](https://github.com/dallascard/krippendorff_alpha)
- [R Implementation of Krippendorff's Alpha](https://rpubs.com/jacoblong/content-analysis-krippendorff-alpha-R)

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
.venv/bin/python3.13
```

**Note:** This project requires Python 3.13 or higher.

Done. You're set up!

---------------

## Project Structure

### Project Structure Overview

The project is structured to ensure **modularity, scalability, and ease of use**. Below is a breakdown of the key components:

- **`src/krippendorff_alpha/`**  
  Contains the core implementation of Krippendorff's Alpha, including modules for preprocessing, reliability computation, and metric calculation.

- **`datasets/`**  
  Contains example datasets in TSV format to demonstrate the usage of Krippendorff's Alpha for different data types (**nominal, ordinal, interval, ratio**).  

- **`notebooks/`**  
  Contains usability examples for the code.

---------------

## Installation

```bash
pip install krippendorff-aleph-alpha
```

Or using UV:

```bash
uv add krippendorff-aleph-alpha
```

## Usage

```python
import pandas as pd
from krippendorff_alpha import compute_alpha
from krippendorff_alpha.schema import ColumnMapping

# Load your data
df = pd.read_csv("your_data.tsv", sep="\t")

# Define column mapping
column_mapping = ColumnMapping(
    text_col="text",
    annotator_cols=["annotator1", "annotator2", "annotator3"]
)

# Compute Krippendorff's alpha
results = compute_alpha(
    df=df,
    data_type="nominal",
    column_mapping=column_mapping
)

print(f"Alpha: {results['alpha']}")
```

## Limitations

1. Krippendorff's alpha is more meaningful for larger data samples, however low agreement can still be a sign of labeling issue. 
2. The code requires at least three annotator columns to compute inter-annotator agreement.
3. The function computes Krippendorff's alpha but does not provide statistical significance (e.g., confidence intervals).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! See the [Contributing.md](krippendorff-aleph-alpha/CONTRIBUTING.md)