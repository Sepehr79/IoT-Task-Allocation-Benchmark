import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from config import NUM_EDGES


# ============================================================
# Configuration
# ============================================================

TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10
RANDOM_STATE = 42

BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.001
L2_ALPHA = 0.0001


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


# ============================================================
# Target normalization
#
# Equivalent to:
#
# TransformedTargetRegressor(
#     regressor=mlp,
#     transformer=StandardScaler()
# )
# ============================================================

target_scaler = StandardScaler()

y_train_scaled = target_scaler.fit_transform(
    y_train
)

y_val_scaled = target_scaler.transform(
    y_val
)


# ============================================================
# L2 Regularization
# ============================================================

l2_reg = tf.keras.regularizers.l2(
    L2_ALPHA
)


# ============================================================
# MLP Model
# ============================================================

inputs = tf.keras.Input(
    shape=(X_train.shape[1],)
)


# Input normalization
x = normalization(inputs)


# Hidden layer 1
x = tf.keras.layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=l2_reg
)(x)


# Hidden layer 2
x = tf.keras.layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=l2_reg
)(x)


# Hidden layer 3
x = tf.keras.layers.Dense(
    64,
    activation="relu",
    kernel_regularizer=l2_reg
)(x)


# Hidden layer 4
x = tf.keras.layers.Dense(
    32,
    activation="relu",
    kernel_regularizer=l2_reg
)(x)


# Hidden layer 5
x = tf.keras.layers.Dense(
    16,
    activation="relu",
    kernel_regularizer=l2_reg
)(x)


# Output layer
outputs = tf.keras.layers.Dense(
    NUM_EDGES,
    activation="linear"
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# Loss
# ============================================================


#loss_fn = combined_loss

def combined_loss(
    y_true,
    y_pred,
    mse_weight=1.0,
    rank_weight=0.5,
    margin=0.0,
    queue_weight_alpha=1.0,   # controls how much extra weight high-delay (queue-dominated) samples get
):
    """
    Combined loss = weighted MSE (emphasizing queue-delay-dominated cases)
                    + pairwise ranking loss (relative order / regret accuracy)

    y_true, y_pred: shape (batch_size, NUM_EDGES)
    """

    # ------------------------------------------------------------
    # Weighted MSE
    #
    # Queue delay is the largest, most variable component of total
    # delay. When true delay is large, it's almost always because
    # queue delay dominates. Weighting those cases more forces the
    # model to fit the queue-dominated regime more accurately,
    # instead of being pulled toward the easy, low-delay bulk.
    # ------------------------------------------------------------
    abs_y_true = tf.abs(y_true)
    mean_abs = tf.reduce_mean(abs_y_true) + 1e-8

    weight = 1.0 + queue_weight_alpha * (abs_y_true / mean_abs)

    weighted_sq_error = weight * tf.square(y_true - y_pred)
    weighted_mse = tf.reduce_mean(weighted_sq_error)

    # ------------------------------------------------------------
    # Pairwise ranking loss (unchanged from before)
    # ------------------------------------------------------------
    true_diff = tf.expand_dims(y_true, axis=2) - tf.expand_dims(y_true, axis=1)
    pred_diff = tf.expand_dims(y_pred, axis=2) - tf.expand_dims(y_pred, axis=1)

    sign_true = tf.sign(true_diff)
    pairwise_hinge = tf.nn.relu(margin - sign_true * pred_diff)

    mask = tf.cast(tf.not_equal(sign_true, 0.0), tf.float32)
    rank_loss = tf.reduce_sum(pairwise_hinge * mask) / (tf.reduce_sum(mask) + 1e-8)

    return mse_weight * weighted_mse + rank_weight * rank_loss

#loss_fn = tf.keras.losses.MeanSquaredError()
loss_fn = lambda y_true, y_pred: combined_loss(
    y_true, y_pred,
    mse_weight=1.0,
    rank_weight=0.5,
    queue_weight_alpha=1.0,   # این رو بین 0.5 تا 2 sweep کنید
)


# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss=loss_fn,
    metrics=["mae"]
)


# ============================================================
# Model summary
# ============================================================

model.summary()


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

history = model.fit(
    X_train,
    y_train_scaled,

    validation_data=(
        X_val,
        y_val_scaled
    ),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# Save model
# ============================================================

model.save(
    "delay_predictor.keras"
)

joblib.dump(
    target_scaler,
    "target_scaler.pkl"
)


# ============================================================
# Prediction
# ============================================================

y_pred_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# Convert predictions back to original scale
#
# Model output:
#       standardized delay
#
# After inverse_transform:
#       delay in seconds
# ============================================================

y_pred = target_scaler.inverse_transform(
    y_pred_scaled
)


# ============================================================
# Show predictions
# ============================================================

print("\nTrue:")
print(y_test[:5])

print("\nPredicted:")
print(y_pred[:5])


# ============================================================
# Best edge
# ============================================================

print("\nTrue best:")
print(
    np.argmin(
        y_test[:5],
        axis=1
    )
)

print("\nPredicted best:")
print(
    np.argmin(
        y_pred[:5],
        axis=1
    )
)


# ============================================================
# Overall metrics
# ============================================================

mse = np.mean(
    (y_test - y_pred) ** 2
)

rmse = np.sqrt(
    mse
)

mae = np.mean(
    np.abs(
        y_test - y_pred
    )
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
            y_test[:, i]
            -
            y_pred[:, i]
        )
    )

    edge_mse = np.mean(
        (
            y_test[:, i]
            -
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
    true_best_edge
    ==
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