import os
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score

# ==================
TRAIN_CSV     = "/Users/teng/Documents/Victoria/ResearchAssistant/2.project/Low-res-less-variables/ice_thickness/dataout/preprocessed_train.csv"
TEST_CSV     = "/Users/teng/Documents/Victoria/ResearchAssistant/2.project/Low-res-less-variables/ice_thickness/dataout/preprocessed_test.csv"
LABEL_COLUMN  = "ice_thickness"
YEAR_COLUMN   = "year"
VALID_SIZE    = 0.2
RANDOM_SEED  = 42

# FOR GA
POP_SIZE      = 20
N_GEN         = 10
CROSSOVER_PROB= 0.8
MUTATION_PROB = 0.1
# =====================

class FeatureSelectionProblem(ElementwiseProblem):
    """
    Define an optimization problem where each decision variable is binary, indicating whether a particular feature is included (1) or excluded (0). The objective of this problem is to minimize the RMSE computed on the validation set.
    """

    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val   = X_val
        self.y_train = y_train
        self.y_val   = y_val
        self.n_labels = X_train.shape[1]

        super().__init__(n_var=self.n_labels, n_obj=1, xl=0, xu=1, type_var=int)


    def _evaluate(self, X, out, *args, **kwargs):
        mask = np.array(X, dtype=bool)  # 1=True select 2=False not select
        if not np.any(mask):
            # If no features selected, assign a high penalty (very large RMSE)
            out["F"] = 1e6
        else:

            X_train_subset = self.X_train[:, mask]
            X_val_subset   = self.X_val[:, mask]
            # Train an MLPRegressor on the selected features
            model = MLPRegressor(hidden_layer_sizes=(100,50), activation='relu', solver='adam', max_iter=500, early_stopping=True, random_state=RANDOM_SEED)
            model.fit(X_train_subset, self.y_train)
            # Predict on validation set and calculate RMSE
            preds = model.predict(X_val_subset)
            rmse = np.sqrt(mean_squared_error(self.y_val, preds))
            out["F"] = rmse


if __name__ == "__main__":
    df_train = pd.read_csv(TRAIN_CSV)
    df_train = df_train.drop(columns=[YEAR_COLUMN])

    X = df_train.drop(columns=[LABEL_COLUMN])
    y = df_train[LABEL_COLUMN]

    feature_names = X.columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X.to_numpy(), y.to_numpy(), test_size=VALID_SIZE, random_state=RANDOM_SEED
    )

    problem = FeatureSelectionProblem(X_train, X_val, y_train, y_val)

    algorithm = GA(
        pop_size=POP_SIZE,
        sampling  = BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=CROSSOVER_PROB),
        mutation=BitflipMutation(prob=MUTATION_PROB),
        eliminate_duplicates=True)

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", N_GEN),
        seed = RANDOM_SEED,
        save_history = True,
        verbose = True
    )

    best_mask = res.X
    if best_mask.ndim > 1:
        best_mask = best_mask[0]
    selected_indices = np.where(best_mask)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    best_rmse = res.F[0] if isinstance(res.F, np.ndarray) else float(res.F)

    print("\n=== GA Feature Selection Results ===")
    print(f"Best RMSE (validation set): {best_rmse:.4f}")
    print(f"Number of selected features: {len(selected_features)} / {X.shape[1]}")
    print("List of selected features:")
    for feat in selected_features:
        print("  -", feat)


    """Testing on TestSet """
    df_test = pd.read_csv(TEST_CSV).drop(columns=[YEAR_COLUMN])
    X_test = df_test[selected_features]
    y_test = df_test[LABEL_COLUMN]
    final_model = MLPRegressor(hidden_layer_sizes=(100,50), activation='relu', solver='adam', max_iter=500, early_stopping=True, random_state=RANDOM_SEED).fit(X[selected_features], y)
    y_pred_test = final_model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2   = r2_score(y_test, y_pred_test)
    print(f"\nTest Set RMSE: {test_rmse:.4f}")
    print(f"\nTest Set r2: {test_r2:.4f}")
