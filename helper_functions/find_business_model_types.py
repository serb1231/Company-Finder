import pandas as pd
import os

if __name__ == "__main__":
    SOURCE_DB = "../data/companies_enhanced.jsonl"

    if not os.path.exists(SOURCE_DB):
        print(f"Error: {SOURCE_DB} not found.")
    else:
        # Load the JSONL file
        df = pd.read_json(SOURCE_DB, lines=True)

        if "business_model" in df.columns:
            # Extract unique values, drop nulls, and sort them alphabetically
            vocab = sorted({bm for row in df["business_model"].dropna() for bm in row})

            print(f"Found {len(vocab)} unique business models:\n")
            for model in vocab:
                print(f"- {model}")
        else:
            print("The column 'business_model' was not found in the file.")