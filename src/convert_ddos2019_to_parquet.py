"""Convert the CIC-DDoS2019 CSV release to Parquet (step 0 of the data pipeline).

This is the conversion that produced the Parquet files read by src/prepare_data.py (folders
train_parquet/ for day 01-12 and test_parquet/ for day 03-11, one Parquet file per CSV file).
It was originally run as a notebook (its saved output lists every converted file); the logic below is
copied from it unchanged. The resulting files are consistent with it: none has the SimillarHTTP column
or any infinite or missing value.

Per CSV file, the data are read in chunks of 100,000 rows. In every chunk:
  * +-inf -> NaN, rows with any NaN are dropped,
  * duplicate rows WITHIN the chunk are dropped; the comparison uses the whole record, including the flow
    ID, addresses, ports and timestamp, so only exact repeats of a full record are removed (duplicates in
    the 76 features plus the original label are removed later, by prepare_data.py),
  * the text column "SimillarHTTP" is dropped.

Usage:
  python src/convert_ddos2019_to_parquet.py --train_csv <.../CSV-01-12> --test_csv <.../CSV-03-11> \
         --out <dir>            # writes <dir>/train_parquet/*.parquet and <dir>/test_parquet/*.parquet
Then run prepare_data.py with --ddos_root <dir>.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(inplace=True)
    chunk.drop_duplicates(inplace=True)
    if "SimillarHTTP" in chunk.columns:
        chunk.drop(columns=["SimillarHTTP"], inplace=True)
    return chunk


def process_csv_to_parquet(csv_path: Path, output_path: Path, chunksize: int = 100_000) -> None:
    writer = None
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        chunk = clean_chunk(chunk)
        if chunk.empty:
            continue
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()
    print(f"Saved: {output_path.name}", flush=True)


def process_folder(input_dir: str, output_dir: str, chunksize: int = 100_000) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for csv_file in Path(input_dir).glob("*.csv"):
        print(f"Processing {csv_file.name}", flush=True)
        process_csv_to_parquet(csv_file, out / f"{csv_file.stem}.parquet", chunksize)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, help="folder with the day 01-12 CSV files (CSV-01-12)")
    ap.add_argument("--test_csv", required=True, help="folder with the day 03-11 CSV files (CSV-03-11)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    process_folder(a.train_csv, str(Path(a.out) / "train_parquet"))
    process_folder(a.test_csv, str(Path(a.out) / "test_parquet"))
