from pathlib import Path
import pandas as pd


def convert_to_csv(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        input_path,
        sep="|",
        engine="python",
        skipinitialspace=True,
    )

    df.to_csv(output_path, index=False)
    print(f"Converted {input_path} -> {output_path} ({len(df):,} rows)")


def main():
    input_file = Path(r"C:\Users\edjwr\Documents\Part 3 Project\Iridis X Data\logs-txt\slurm_accounting_2025-07-17.txt")
    output_file = Path(r"C:\Users\edjwr\Documents\Part 3 Project\Iridis X Data\test\slurm_2025_07_17.csv")
    convert_to_csv(input_file, output_file)


if __name__ == "__main__":
    main()
