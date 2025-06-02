import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, r2_score

# =====================
LABEL_COLUMN  = "ice_velocity"
OUT_DICT     = f"/Users/teng/Documents/Victoria/ResearchAssistant/2.project/Low-res-more-variables/{LABEL_COLUMN}"
INPUT_FILE   = f"{OUT_DICT}/{LABEL_COLUMN}_all_years.csv"
YEAR_COL     = "year"
TRAIN_RATIO  = 0.8
DATA_OUT_DIR     = f"{OUT_DICT}/dataout"
TRAIN_OUT    = f"{DATA_OUT_DIR}/train.csv"
TEST_OUT     = f"{DATA_OUT_DIR}/test.csv"
PREPROCESSED_TRAIN_OUT = f"{DATA_OUT_DIR}/preprocessed_train.csv"
PREPROCESSED_TEST_OUT = f"{DATA_OUT_DIR}/preprocessed_test.csv"
PREDICT_OUTPUT_DIR   = f"{OUT_DICT}/model_results"
RANDOM_SEED = 42
# =====================

def split_data():

    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(by=YEAR_COL)

    years = sorted(df[YEAR_COL].unique())
    n_years = len(years)

    n_train = int(n_years * TRAIN_RATIO)
    n_train = max(1, min(n_train, n_years - 1))

    train_years = set(years[:n_train])
    test_years = set(years[n_train:])

    train = df[df[YEAR_COL].isin(train_years)].copy()
    test  = df[df[YEAR_COL].isin(test_years)].copy()

    train.to_csv(TRAIN_OUT, index=False)
    test .to_csv(TEST_OUT,  index=False)

    print("total length = ", len(df))
    print("train length = ", len(train))
    print("test length = ", len(test))

    return train, test

def data_overview(data, file_name):
    print("==========data.head==========")
    print(data.head())
    print("==========data.info()==========")
    print(data.info())
    print("==========data.describe()==========")
    print(data.describe())
    data.describe().T.to_csv(f"{OUT_DICT}/{file_name}")

def missing_value_check(data):
    print("==========Number of missing values==========")
    print(data.isnull().sum())
    print("==========Proportion of missing values==========")
    print(data.isnull().mean())

def variable_distribution(data, folder_name):

    num_cols = data.select_dtypes(include=np.number).columns.tolist()

    exclude = {"x_coordinate", "y_coordinate", "year"}
    num_cols = [c for c in num_cols if c not in exclude]
    output_img_dir = os.path.join(OUT_DICT, folder_name)
    os.makedirs(output_img_dir, exist_ok=True)

    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
        axes[0].hist(data[col].dropna(), bins=30)
        axes[0].set_title(f"{col} distribution")
        axes[1].boxplot(data[col].dropna(), vert=True)
        axes[1].set_title(f"{col} box plot")
        plt.suptitle(col, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        img_path = os.path.join(output_img_dir, f"{col}_dist_box.png")
        fig.savefig(img_path, bbox_inches='tight')
        plt.close(fig)


def correlation_heat_map(data, file_name):
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if 'year' in num_cols:
        num_cols.remove('year')

    corr = data[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols, fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.colorbar(im, ax=ax)
    plt.title("Numeric Feature Correlation Matrix", fontsize=12)

    plt.savefig(f"{OUT_DICT}/{file_name}", dpi=300, bbox_inches="tight")
    plt.close(fig)

def time_trend(data):
    yearly = data.groupby("year")[["precipitation", "air_temperature", "ocean_temperature"]].mean()

    plt.figure(figsize=(8, 4))
    yearly.plot(marker="o")
    plt.title("Annual Mean Trend")
    plt.ylabel("Mean Value")
    plt.xlabel("Year")
    plt.grid(True)

    plt.savefig(f"{OUT_DICT}/yearly_trend.png", dpi=300, bbox_inches="tight")
    plt.close()

def geographic_distribution(data):
    output_img_dir = os.path.join(OUT_DICT, "geographic_distribution")
    os.makedirs(output_img_dir, exist_ok=True)

    for year, df_year in data.groupby("year"):
        if df_year.empty:
            continue

        plt.scatter(
            df_year["x_coordinate"],
            df_year["y_coordinate"],
            c=df_year[LABEL_COLUMN],
            s=5,
            cmap="viridis",
        )
        plt.colorbar(label=LABEL_COLUMN)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{LABEL_COLUMN} Distribution, Year = {year}")

        out_path = os.path.join(output_img_dir, f"geo_dist_{year}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

def outlier_data(data):
    Q1 = data["precipitation"].quantile(0.25)
    Q3 = data["precipitation"].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[
        (data["precipitation"] < Q1 - 1.5 * IQR) |
        (data["precipitation"] > Q3 + 1.5 * IQR)
    ]
    print("Number of precipitation outliers:", len(outliers))

def preprocess_data(train_df, test_df, impute_strategy: str = 'median'):
    all_numeric = train_df.select_dtypes(include='number').columns.tolist()

    if LABEL_COLUMN in all_numeric:
        numeric_features = [col for col in all_numeric if col != LABEL_COLUMN]
    else:
        numeric_features = all_numeric.copy()

    # Outlier Handling
    train_clipped = train_df[numeric_features].copy()
    test_clipped  = test_df[numeric_features].copy()

    bounds = {}
    for col in numeric_features:
        Q1 = train_clipped[col].quantile(0.25)
        Q3 = train_clipped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        bounds[col] = (lower, upper)

        train_clipped[col] = train_clipped[col].clip(lower, upper)
        test_clipped[col]  = test_clipped[col].clip(lower, upper)

    # for Imputation
    imputer = SimpleImputer(strategy=impute_strategy)
    train_imputed = imputer.fit_transform(train_clipped)
    test_imputed  = imputer.transform(test_clipped)

    # standardization
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled  = scaler.transform(test_imputed)

    train_processed = train_df.copy()
    test_processed  = test_df.copy()
    train_processed[numeric_features] = train_scaled
    test_processed[numeric_features]  = test_scaled

    train_processed.to_csv(PREPROCESSED_TRAIN_OUT, index=False)
    test_processed .to_csv(PREPROCESSED_TEST_OUT,  index=False)

    return train_processed, test_processed

def split_X_y(train, test):
    X_train = train.drop([LABEL_COLUMN, "year"], axis=1)
    y_train = train[LABEL_COLUMN]
    X_test  = test .drop([LABEL_COLUMN, "year"], axis=1)
    y_test  = test [LABEL_COLUMN]
    return X_train, y_train, X_test, y_test

def predict_and_evaluate(model, name, X_train, y_train, X_test, y_test, output_dir, predict_label):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"{name:20s} → RMSE = {rmse:.3f}, R² = {r2:.3f}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, s=10, alpha=0.6)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val],
             [min_val, max_val],
             "k--", lw=1)
    plt.xlabel(f"True {predict_label}")
    plt.ylabel(f"Predicted {predict_label}")
    plt.title(f"{name} Predictions\nRMSE={rmse:.2f}, R²={r2:.2f}")
    plt.tight_layout()

    fname = os.path.join(output_dir, f"{name}.png")
    plt.savefig(fname)
    plt.close()

def training_testing_on_diff_models(X_train, y_train, X_test, y_test):

    models = [
        (LinearRegression(), "LinearRegression"),
        (Lasso(alpha=0.1, max_iter=10000), "Lasso_alpha0.1"),
        (SVR(kernel='rbf', C=1.0, epsilon=0.1),"SVR_RBF"),
        (DecisionTreeRegressor(max_depth=10, random_state=RANDOM_SEED), "DecisionTree"),
        (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_SEED, n_jobs=-1), "RandomForest"),
        (MLPRegressor(hidden_layer_sizes=(100,50), activation='relu', solver='adam', max_iter=500, early_stopping=True, random_state=RANDOM_SEED), "MLPRegressor"),
    ]

    os.makedirs(PREDICT_OUTPUT_DIR, exist_ok=True)

    for model, name in models:
        predict_and_evaluate(model, name,
                             X_train, y_train,
                             X_test,  y_test,
                             PREDICT_OUTPUT_DIR, LABEL_COLUMN)


if __name__ == "__main__":
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    train_set, test_set = split_data()

    """EDA-raw data"""
    # data_overview(train_set, "data_describe_summary.csv")
    # missing_value_check(train_set)
    # variable_distribution(train_set,"distribution_plot")
    # correlation_heat_map(train_set, "correlation_matrix.png")
    # time_trend(train_set)
    # geographic_distribution(train_set)
    # outlier_data(train_set)

    train_processed, test_processed = preprocess_data(train_set, test_set)

    """EDA-processed data"""
    data_overview(train_processed, "processed_data_describe_summary.csv")
    # missing_value_check(train_processed)
    variable_distribution(train_processed, "PreProcessed_distribution_plot")
    correlation_heat_map(train_processed, "PreProcessed_correlation_matrix.png")

    # predict and evaluate
    X_train, y_train, X_test, y_test = split_X_y(train_processed, test_processed)
    training_testing_on_diff_models(X_train, y_train, X_test, y_test)
