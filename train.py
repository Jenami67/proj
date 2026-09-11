"""Training script (Section 3.1.6): Adam optimizer (lr=0.001, beta1=0.9,
beta2=0.999), categorical cross-entropy with class weighting."""

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

import config
from model import build_hybrid_model


def load_split(split_name):
    mel = np.load(config.SPLITS_DIR / f"{split_name}_mel.npy")
    mfcc = np.load(config.SPLITS_DIR / f"{split_name}_mfcc.npy")
    y = np.load(config.SPLITS_DIR / f"{split_name}_y.npy")
    return mel, mfcc, y


def main():
    train_mel, train_mfcc, y_train = load_split("train")
    val_mel, val_mfcc, y_val = load_split("val")

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=2)

    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights_arr))
    print("Class weights:", class_weights)

    model = build_hybrid_model(
        mel_shape=train_mel.shape[1:], mfcc_shape=train_mfcc.shape[1:]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE, beta_1=0.9, beta_2=0.999
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODELS_DIR / "hybrid_cnn_rnn_best.keras"),
            monitor="val_loss", save_best_only=True,
        ),
    ]

    model.fit(
        [train_mel, train_mfcc], y_train_cat,
        validation_data=([val_mel, val_mfcc], y_val_cat),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save(config.MODELS_DIR / "hybrid_cnn_rnn_final.keras")
    print("Training complete. Model saved to", config.MODELS_DIR)


if __name__ == "__main__":
    main()
