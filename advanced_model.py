import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
import joblib
from sklearn.model_selection import train_test_split
from config import NUM_EDGES


# ============================================================
# Configuration
# ============================================================

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(
    "allocation_dataset.csv"
)

print(f"Dataset size: {len(df)} rows")


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
].values.astype(np.float32)

y = df[
    target_columns
].values.astype(np.float32)


# ============================================================
# Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)


# ============================================================
# Train / validation split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)


print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples:    {len(X_test)}")


# ============================================================
# Input normalization
# ============================================================

normalization = tf.keras.layers.Normalization()

normalization.adapt(X_train)


target_scaler = StandardScaler()

y_train_scaled = target_scaler.fit_transform(y_train)
y_val_scaled = target_scaler.transform(y_val)
y_test_scaled = target_scaler.transform(y_test)

# ============================================================
# MLP
# ============================================================

inputs = tf.keras.Input(
    shape=(X_train.shape[1],)
)

x = normalization(inputs)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)

x = tf.keras.layers.Dense(
    64,
    activation="relu"
)(x)

x = tf.keras.layers.Dense(
    32,
    activation="relu"
)(x)

x = tf.keras.layers.Dense(
    16,
    activation="relu"
)(x)

outputs = tf.keras.layers.Dense(
    NUM_EDGES
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# Loss
# ============================================================

loss_fn = tf.keras.losses.MeanSquaredError()


# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=loss_fn,
    metrics=["mae"]
)


# ============================================================
# Early stopping
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)


# ============================================================
# Train
# ============================================================

model.fit(
    X_train,
    y_train_scaled,
    validation_data=(X_val, y_val_scaled),
    epochs=500,
    batch_size=64,
    callbacks=[early_stopping]
)


# ============================================================
# Save complete model
# ============================================================

model.save(
    "delay_predictor.keras"
)

joblib.dump(
    target_scaler,
    "target_scaler.pkl"
)
# ============================================================
# Evaluation on Test Set
# ============================================================

y_pred_scaled = model.predict(
    X_test,
    verbose=0
)


# Convert predictions back to original delay scale
y_pred = target_scaler.inverse_transform(
    y_pred_scaled
)


# ============================================================
# Overall metrics
# ============================================================

mse = np.mean(
    (y_test - y_pred) ** 2
)

rmse = np.sqrt(mse)

mae = np.mean(
    np.abs(y_test - y_pred)
)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(
    f"Test MSE:  {mse:.6f}"
)

print(
    f"Test RMSE: {rmse:.6f} s"
)

print(
    f"Test MAE:  {mae:.6f} s"
)


# ============================================================
# Evaluation per Edge
# ============================================================

print("\nPer-Edge Evaluation:")
print("-" * 60)

for i in range(NUM_EDGES):

    edge_mae = np.mean(
        np.abs(
            y_test[:, i] -
            y_pred[:, i]
        )
    )

    edge_mse = np.mean(
        (
            y_test[:, i] -
            y_pred[:, i]
        ) ** 2
    )

    edge_rmse = np.sqrt(
        edge_mse
    )

    print(
        f"Edge {i}: "
        f"MAE = {edge_mae:.6f} s, "
        f"RMSE = {edge_rmse:.6f} s"
    )


# ============================================================
# Best Edge Accuracy
# ============================================================

true_best_edge = np.argmin(
    y_test,
    axis=1
)

predicted_best_edge = np.argmin(
    y_pred,
    axis=1
)


best_edge_accuracy = np.mean(
    true_best_edge ==
    predicted_best_edge
)


print("\n" + "-" * 60)

print(
    f"Best-Edge Selection Accuracy: "
    f"{best_edge_accuracy * 100:.2f}%"
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
    predicted_best_edge
]

regret = (
    selected_delay -
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
print("=" * 60)
print("ALLOCATION RESULTS")
print("=" * 60)

print(
    f"Average regret: "
    f"{average_regret:.6f} s"
)

print(
    f"Median regret: "
    f"{median_regret:.6f} s"
)

print(
    f"95th percentile regret: "
    f"{p95_regret:.6f} s"
)

print(
    f"Maximum regret: "
    f"{maximum_regret:.6f} s"
)

print("=" * 60)