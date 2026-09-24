import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import *


# ============================================================
# Configuration
# ============================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(
    "allocation_dataset.csv"
)

print(
    f"Dataset size: {len(df)} rows"
)


# ============================================================
# Input features
# ============================================================

feature_columns = (
    [
        "processing_capcacity",
        "data_size"
    ]

    +

    [
        f"queue_{i}"
        for i in range(NUM_EDGES)
    ]

    +

    [
        f"last_allocation_{i}"
        for i in range(NUM_EDGES)
    ]

    +

    [
        f"rate_{i}"
        for i in range(NUM_EDGES)
    ]
)


# ============================================================
# Target variables
# ============================================================

target_columns = [
    f"delay_{i}"
    for i in range(NUM_EDGES)
]


# ============================================================
# Create X and y
# ============================================================

X = df[
    feature_columns
].values

y = df[
    target_columns
].values


# ============================================================
# Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples:  {len(X_test)}"
)


# ============================================================
# MLP
# ============================================================

mlp = MLPRegressor(

    hidden_layer_sizes=(
        64,
        64,
        32,
        32,
        16
    ),

    activation="relu",

    solver="adam",

    learning_rate_init=0.001,

    max_iter=500,

    batch_size=64,

    early_stopping=True,

    validation_fraction=0.15,

    n_iter_no_change=20,

    random_state=RANDOM_STATE,
)


# ============================================================
# Input scaler + MLP
# ============================================================

model = Pipeline(
    steps=[

        (
            "input_scaler",
            StandardScaler()
        ),

        (
            "regressor",
            TransformedTargetRegressor(
                regressor=mlp,
                transformer=StandardScaler()
            )
        )
    ]
)


# ============================================================
# Train
# ============================================================

model.fit(
    X_train,
    y_train
)

# ============================================================
# Prediction
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# Regression metrics
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


print()
print("=" * 55)
print("REGRESSION RESULTS")
print("=" * 55)

print(
    f"MAE:  {mae:.4f} s"
)

print(
    f"RMSE: {rmse:.4f} s"
)


# ============================================================
# Best-edge accuracy
# ============================================================

actual_best = np.argmin(
    y_test,
    axis=1
)

predicted_best = np.argmin(
    y_pred,
    axis=1
)


accuracy = np.mean(
    actual_best == predicted_best
)


print(
    f"Best-edge accuracy: "
    f"{accuracy:.4f}"
)


# ============================================================
# Regret
# ============================================================

optimal_delay = np.min(
    y_test,
    axis=1
)


selected_delay = y_test[
    np.arange(len(y_test)),
    predicted_best
]


regret = (
    selected_delay
    -
    optimal_delay
)


average_regret = np.mean(
    regret
)

median_regret = np.median(
    regret
)

p95_regret = np.percentile(
    regret,
    95
)

maximum_regret = np.max(
    regret
)


print()
print("=" * 55)
print("ALLOCATION RESULTS")
print("=" * 55)

print(
    f"Average regret: "
    f"{average_regret:.4f} s"
)

print(
    f"Median regret: "
    f"{median_regret:.4f} s"
)

print(
    f"95th percentile regret: "
    f"{p95_regret:.4f} s"
)

print(
    f"Maximum regret: "
    f"{maximum_regret:.4f} s"
)


# ============================================================
# Save complete pipeline
# ============================================================

joblib.dump(
    model,
    "delay_prediction_pipeline.joblib"
)

print()
print(
    "Model saved as "
    "delay_prediction_pipeline.joblib"
)