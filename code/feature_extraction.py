import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_and_clean_file(path, columns, fill_value=9.96920996839e+36):
    """Read file, assign column names, replace fill_value→NaN and drop NaN rows."""
    df = pd.read_csv(path, sep="\t", header=None)
    if len(columns) == df.shape[1]:
        df.columns = columns
    return df.replace(fill_value, np.nan).dropna()

def reconstruct_dataset(df, label_column="ice_thickness", center_features=["x_coordinate", "y_coordinate", "precipitation", "air_temperature", "ocean_temperature"]):
    """
    Reconstructs the dataset such that each row corresponds to a grid cell (with full 8 neighbors).
    The features for each row include the same three/four variables for each of the eight surrounding neighbors (ordered as:
          top-left, top, top-right, left, right, bottom-left, bottom, bottom-right).
    """
    # Determine the unique x and y_coordinates to infer the grid shape.
    x_unique = np.sort(df["x_coordinate"].unique())
    y_unique = np.sort(df["y_coordinate"].unique())
    nx = len(x_unique)
    ny = len(y_unique)

    x_to_idx = {x_val: idx for idx, x_val in enumerate(x_unique)}
    y_to_idx = {y_val: idx for idx, y_val in enumerate(y_unique)}

    data_map = {(row["x_coordinate"], row["y_coordinate"]): row
                for _, row in df.iterrows()}

    # Offsets for 8 neighbors in the grid: (row_offset, col_offset)
    neighbor_offsets = [(-1,  1), ( 0,  1), ( 1,  1),
                        (-1,  0),           ( 1,  0),
                        (-1, -1), ( 0, -1), ( 1, -1)]

    # Create names for neighbor features.
    position_names = {  (-1,  1): "upper_left",
                        ( 0,  1): "top",
                        ( 1,  1): "upper_right",
                        (-1,  0): "left",
                        ( 1,  0): "right",
                        (-1, -1): "lower_left",
                        ( 0, -1): "bottom",
                        ( 1, -1): "lower_right"}

    # Feature names for the center cell
    base_feats = center_features[2:]
    neighbor_feature_names = []
    for offset in neighbor_offsets:
        pos_name = position_names[offset]
        for feat in base_feats:
            neighbor_feature_names.append(f"{pos_name}_{feat}")

    # Final column names for reconstructed data: features and then label.
    final_columns = center_features + neighbor_feature_names + [label_column]

    new_rows = []

    # Loop over valid grid indices
    for _, row in df.iterrows():
        cx, cy = row["x_coordinate"], row["y_coordinate"]
        if cx in x_to_idx and cy in y_to_idx:
            ix = x_to_idx[cx]
            iy = y_to_idx[cy]
        else:
            raise KeyError(f"can't find coordinate ({cx}, {cy})")

        center_vals = [row[c] for c in center_features]

        nbr_vals = []
        for dx, dy in neighbor_offsets:
            nix = ix + dx   # horizontal offset
            niy = iy + dy   # vertical offset

            if 0 <= nix < nx and 0 <= niy < ny and (x_unique[nix], y_unique[niy]) in data_map:
                nbr = data_map[(x_unique[nix], y_unique[niy])]
                for feat in base_feats:
                    nbr_vals.append(nbr[feat])
            else:
                # Out of bounds or missing data → add 0
                nbr_vals.extend([0.0] * len(base_feats))

        label_val = row[label_column]
        new_rows.append(center_vals + nbr_vals + [label_val])

    return pd.DataFrame(new_rows, columns=final_columns)

def add_historical_features(current_year, df, hist_maps, center_features, label_column="ice_thickness"):
    """
    Given a DataFrame `df` with 'x_coordinate' and 'y_coordinate',
    and a dict hist_maps: year -> {(x,y) -> row},
    append for each year in hist_maps three/four new columns:
      (bed_rock_elevation_<year>), precipitation_<year>, air_temperature_<year>, ocean_temperature_<year>
    missing lookups are zero‑filled.
    """

    base_feats = center_features[2:]
    zero_rec = {feat: 0.0 for feat in base_feats}

    label_idx = df.columns.get_loc(label_column)

    for year in sorted(hist_maps):
        n = current_year - year
        suffix = f"{n}_year_ago" if n == 1 else f"{n}_years_ago"
        for feat in base_feats:
            col_name = f"{feat}_{suffix}"

            ser = df.apply(
                lambda r, m=hist_maps[year], f=feat:
                    m.get((r["x_coordinate"], r["y_coordinate"]), zero_rec)[f],
                axis=1
            )

            df.insert(label_idx, col_name, ser)

            label_idx += 1

    return df

def read_and_reconstruct_data(folder, cols, center_features, k=3, filename="vars-*.txt", year_match=r"vars-(\d{4})\.txt$", label_column="ice_thickness"):

    output_dir = os.path.join(folder, label_column)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{label_column}_all_years.csv")

    if os.path.exists(output_file):
        os.remove(output_file)

    files = glob.glob(os.path.join(folder, filename))
    files = sorted(files, key=lambda fn: int(re.search(year_match, fn).group(1)))
    if (len(files) <= 0):
        raise IndexError(f"no files")

    first_file = True #Only the first file writes the header

    total_data_len = 0

    # 1. Read the nth file, get its year
    for fp in files:
        # extract the 4‑digit year from the filename
        match = re.search(year_match, os.path.basename(fp))
        if not match:
            raise ValueError(f"Cannot parse year from {fp}")
        year = int(match.group(1))
        df_cur = read_and_clean_file(fp, cols)

        data = {(r["x_coordinate"], r["y_coordinate"]): r for _, r in df_cur.iterrows()}

        total_data_len = total_data_len + len(df_cur)

        base_feats = center_features[2:]
        # 2. Build historical maps for past k years
        hist_maps = {}
        for i in range(1, k+1):
            y = year - i
            year_pattern = filename.replace("*", str(y))
            y_fp = os.path.join(folder, year_pattern)
            if os.path.isfile(y_fp):
                df_h = read_and_clean_file(y_fp, cols)
                hist_maps[y] = {
                    (r["x_coordinate"], r["y_coordinate"]): r
                    for _, r in df_h.iterrows()
                }
            else:
                # Missing year → map every current coord to zeros
                zero_rec = {feat: 0.0 for feat in base_feats}
                hist_maps[y] = {
                    coord: zero_rec.copy()
                    for coord in data.keys()
                }


        # 3) Reconstruct current+neighbours dataset
        recon_df = reconstruct_dataset(df_cur, label_column, center_features)

        # 4) Add historical features
        recon_with_history = add_historical_features(year, recon_df, hist_maps, center_features, label_column)

        recon_with_history.insert(0, "year", year)

        if first_file:
            recon_with_history.to_csv(output_file, index=False, mode="w", encoding="utf-8")
            first_file = False
        else:
            recon_with_history.to_csv(output_file, index=False, mode="a", header=False, encoding="utf-8")

        # plot_data(df_clean, x_col="x_coordinate", y_col="y_coordinate", label_col="ice_thickness")
    print("total_data_len = ", total_data_len)

def plot_data(df, x_col="x_coordinate", y_col="y_coordinate", label_col="ice_thickness"):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df[x_col], df[y_col], c=df[label_col], cmap="viridis")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{label_col} over {x_col} and {y_col}")
    cbar = plt.colorbar(scatter)
    cbar.set_label(label_col)
    plt.show()

# Example usage
if __name__ == "__main__":


    label_column="ice_thickness"  #Selected label
    past_years = 3         # number of past years

    # Change the folder where the raw data files are stored
    folder = "/Users/teng/Documents/Victoria/ResearchAssistant/2.project/Low-res-more-variables"

    # Configurable column names list (order must match the file's data columns)
    ### For res-less-variables：
    # cols = [
    #     "x_coordinate",
    #     "y_coordinate",
    #     "ice_thickness",
    #     "ice_velocity",
    #     "ice_mask",
    #     "precipitation",
    #     "air_temperature",
    #     "ocean_temperature"
    # ]
    #
    # center_features=["x_coordinate", "y_coordinate", "precipitation", "air_temperature", "ocean_temperature"]
    #
    # filename="vars-*.txt"
    # year_match=r"vars-(\d{4})\.txt$"

    #### For res-more-variables：
    cols = [
        "x_coordinate",
        "y_coordinate",
        "bed_rock_elevation",
        "ice_thickness",
        "ice_velocity",
        "ice_mask",
        "precipitation",
        "air_temperature",
        "ocean_temperature"
    ]

    center_features=["x_coordinate", "y_coordinate", "bed_rock_elevation", "precipitation", "air_temperature", "ocean_temperature"]
    # #
    # filename="vars-*.txt"
    # year_match=r"vars-(\d{4})\.txt$"
    filename    = "vars-*-lowRes.txt"
    year_match  = r"vars-(\d{4})-lowRes\.txt$"

    read_and_reconstruct_data(folder, cols, center_features, past_years, filename, year_match, label_column)
