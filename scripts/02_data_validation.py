import argparse
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema


# ---------------------------------------------------------------------------
# Schema definition
# Check all columns: 
#   Check 1: all must be float
#   Check 2: all must be non-null
#   Check 3: all must be within plausible ranges
#   Check 4: all must have correct column names
#   Check 5: check if target variable follows expected distribution
#   Check 6: check if there is any column with a single value
# ---------------------------------------------------------------------------
schema = DataFrameSchema(
    columns={
        "fixed_acidity": Column(
            float,
            checks=Check.in_range(3.0, 20.0),
            nullable=False,
            description="Tartaric acid concentration (g/dm^3)",
        ),
        "volatile_acidity": Column(
            float,
            checks=Check.in_range(0.0, 2.0),
            nullable=False,
            description="Acetic acid concentration (g/dm^3)",
        ),
        "citric_acid": Column(
            float,
            checks=Check.in_range(0.0, 2.0),
            nullable=False,
            description="Citric acid concentration (g/dm^3)",
        ),
        "residual_sugar": Column(
            float,
            checks=Check.in_range(0.0, 70.0),
            nullable=False,
            description="Residual sugar (g/dm^3)",
        ),
        "chlorides": Column(
            float,
            checks=Check.in_range(0.0, 0.7),
            nullable=False,
            description="Sodium chloride concentration (g/dm^3)",
        ),
        "free_sulfur_dioxide": Column(
            float,
            checks=Check.in_range(0.0, 300.0),
            nullable=False,
            description="Free SO2 (mg/dm^3)",
        ),
        "total_sulfur_dioxide": Column(
            float,
            checks=Check.in_range(0.0, 500.0),
            nullable=False,
            description="Total SO2 (mg/dm^3)",
        ),
        "density": Column(
            float,
            checks=Check.in_range(0.95, 1.05),
            nullable=False,
            description="Density (g/cm^3)",
        ),
        "pH": Column(
            float,
            checks=Check.in_range(2.5, 4.5),
            nullable=False,
            description="pH level",
        ),
        "sulphates": Column(
            float,
            checks=Check.in_range(0.0, 2.5),
            nullable=False,
            description="Potassium sulphate (g/dm^3)",
        ),
        "alcohol": Column(
            float,
            checks=Check.in_range(7.0, 16.0),
            nullable=False,
            description="Alcohol content (% by volume)",
        ),
        "quality": Column(
            int,
            checks=[
                Check.in_range(0, 10),
                Check(
                    lambda s: s.nunique() >= 3,
                    error="Target 'quality' has fewer than 3 distinct values -- "
                          "distribution may be degenerate.",
                ),
            ],
            nullable=False,
            description="Sensory quality score (0-10)",
        ),
    },
    checks=[
        Check(
            lambda df: (df.nunique() > 1).all(),
            error="One or more columns contain only a single unique value.",
        ),
    ],
    coerce=False,   
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the wine quality raw dataset")
    parser.add_argument("--datapath", help="Path to the raw CSV file")
    args = parser.parse_args()
    
    # Check 7: try to read file as a csv and test if there is any load error or wrong file type
    try:
        df = pd.read_csv(args.datapath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {args.data_path}")
    except Exception as e:
        raise ValueError(f"Could not read file as CSV: {e}") from e

    # Check 8: no anomalous correlations between target and features
    target_corr = df.corr(numeric_only=True)['quality'].drop('quality').abs()
    anomalous = target_corr[target_corr > 0.9]

    if not anomalous.empty:
        print(f"WARNING: Anomalous target correlations found:\n{anomalous.to_string()}")

    # Run schema validation; each checks involved are defined in schema definition and specification
    try:
        schema.validate(df, lazy=True)
        print("OK: All schema checks passed.")
    except pa.errors.SchemaErrors as e:
        print("\nVALIDATION FAILED -- errors found:\n")
        # failure_cases is a tidy DataFrame summarising every violation
        print(e.failure_cases.to_string(index=False))