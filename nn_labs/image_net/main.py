from contextlib import suppress
from pathlib import Path

import wandb
import pickle

from nn_labs.image_net.load import load_fashion_mnist
from nn_labs.nn.network import AutoEncoderNetwork

from tqdm import trange

from nn_labs.nn.optimizers import PhilipOptimizer2
from nn_labs.nn.perf import MSELoss


EPOCHS = 200
BATCH_SIZE = 4086
LEARNING_RATE = 10e-3
OUTPUT_DIR = Path(__file__).parent / "models"


wandb_run = wandb.init(
    project="fashion-mnist-autoencoder",
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "764x20",
        "dataset": "Fashion mnist",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
)


(X_train, Y_train), (X_test, Y_test) = load_fashion_mnist()


network = AutoEncoderNetwork(
    input_dim=784,
    encoding_steps=1,
    encoded_dim=20,
)

print(network)


for layer in network.layers[1:-1]:
    layer.l2_bias = 5e-3
    layer.l2_weight = 5e-3


batch_count = X_train.shape[0] // BATCH_SIZE
loss_fn = MSELoss()
optimizer = PhilipOptimizer2(network, learning_rate=LEARNING_RATE)

with suppress(KeyboardInterrupt):
    for epoch in range(EPOCHS):
        bar = trange(batch_count)

        for batch_no in bar:
            start_idx = batch_no * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE

            x = X_train[start_idx:end_idx]
            y = Y_train[start_idx:end_idx]

            prediction = network.forward(x)

            loss = loss_fn.calculate(prediction, x)

            loss_fn.backward(prediction, x)
            network.backwards(loss_fn.d_inputs)

            optimizer.step()

            if not batch_no % (batch_count // 10):
                validation_prediction = network.forward(X_test)
                validation_loss = loss_fn.calculate(validation_prediction, X_test)

                wandb.log({"loss": loss, "val-loss": validation_loss, "epoch": epoch})
                bar.set_description(
                    f"Epoch: {epoch + 1}/{EPOCHS} batch: {batch_no} / "
                    f"{batch_count}. Loss: {loss:.4f} Validation loss: {validation_loss:.4f}"
                )


pickle.dump(network, open(OUTPUT_DIR / f"{wandb_run.name}", "wb"))
wandb.finish()
