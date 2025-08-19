import pyarrow as pa
import pyarrow.parquet as pq


def read_parquet_output_files(file_names):
    for f in file_names:
        try:
            table = pq.read_table(f)
        except pa.ArrowInvalid:
            print(f"Reading of file {f} failed. Skipping it.")
            continue

        yield int(f.parent.name), table.flatten().flatten()
