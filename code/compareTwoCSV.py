import csv

def compare_csv_positions(file1, file2, delimiter=',', encoding='utf-8'):
    """
    Compare the values of all cells (including header row) at the same positions in two CSV files.
    Returns a list of differences: [(row_index, col_index, val1, val2), ...]
    Both row_index and col_index start from 0.
    """
    # Read files as lists of rows (each row is a list of strings)
    with open(file1, newline='', encoding=encoding) as f1, \
         open(file2, newline='', encoding=encoding) as f2:
        reader1 = csv.reader(f1, delimiter=delimiter)
        reader2 = csv.reader(f2, delimiter=delimiter)
        rows1 = list(reader1)
        rows2 = list(reader2)

    max_rows = max(len(rows1), len(rows2))
    max_cols = max(
        max((len(r) for r in rows1), default=0),
        max((len(r) for r in rows2), default=0)
    )

    diffs = []
    for i in range(max_rows):
        for j in range(max_cols):
            # Retrieve values at the corresponding position; treat out-of-bounds as None
            val1 = rows1[i][j] if i < len(rows1) and j < len(rows1[i]) else None
            val2 = rows2[i][j] if i < len(rows2) and j < len(rows2[i]) else None
            if val1 != val2:
                diffs.append((i, j, val1, val2))

    return diffs

if __name__ == "__main__":
    file1 = "/Users/teng/Documents/Victoria/ResearchAssistant/2.project/Low-res-less-variables/ice_velocity/ice_velocity_all_years.csv"
    file2 = "/Users/teng/Documents/Victoria/ResearchAssistant/2.project/Low-res-less-variables/ice_velocity/ice_velocity_all_years_1.csv"

    differences = compare_csv_positions(file1, file2)
    if not differences:
        print("The two files have identical values at all positions.")
    else:
        print("Differences found:")
        for row, col, v1, v2 in differences:
            print(f"  Row {row} Column {col}: {v1!r} != {v2!r}")
