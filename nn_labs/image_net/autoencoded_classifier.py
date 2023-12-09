from pathlib import Path

import wandb
import pickle

from nn_labs.image_net.load import load_fashion_mnist
from nn_labs.nn.network import AutoEncoderNetwork, FFDenseNetwork

from tqdm import trange

from nn_labs.nn.optimizers import PhilipOptimizer2
from nn_labs.nn.perf import CategoricalCrossEntropyLoss, Accuracy
from nn_labs.nn.regularizers import L1Regularizer, L2Regularizer

EPOCHS = 500
BATCH_SIZE = 1000
LEARNING_RATE = 10e-3 * 2
OUTPUT_DIR = Path(__file__).parent / "models"


wandb_run = wandb.init(
    project="fashion-mnist-classifier-encoded",
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "784x20x10",
        "dataset": "Fashion mnist",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
)


(X_train, Y_train), (X_test, Y_test) = load_fashion_mnist()

autoencoder: AutoEncoderNetwork = pickle.load(
    open("/Users/philip/code/uni/nn/nn_labs/image_net/models/fresh-morning-10", "rb")
)
encoder = autoencoder.as_encoder()

network = FFDenseNetwork(
    input_dim=784,
    hidden_dim=20,
    output_dim=10,
    number_of_hidden_layers=2,
)

print(network)

# X_train = encoder.forward(X_train)
# X_test = encoder.forward(X_test)

print(X_test.shape)

batch_count = X_train.shape[0] // BATCH_SIZE
loss_fn = CategoricalCrossEntropyLoss()
acc_fn = Accuracy()
optimizer = PhilipOptimizer2(network, learning_rate=LEARNING_RATE)
l1 = L1Regularizer(network)
l2 = L2Regularizer(network)

in_loss = []
out_loss = []

for epoch in range(EPOCHS):
    bar = trange(batch_count)

    for batch_no in bar:
        start_idx = batch_no * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        x = X_train[start_idx:end_idx]
        y = Y_train[start_idx:end_idx]

        prediction = network.forward(x)

        loss = loss_fn.calculate(prediction, y)
        loss += l1.forward()
        loss += l2.forward()

        loss_fn.backward(prediction, y)
        network.backwards(loss_fn.d_inputs)

        optimizer.step()

        if not batch_no % (batch_count // 10):
            validation_prediction = network.forward(X_test)
            validation_loss = loss_fn.calculate(validation_prediction, Y_test)
            acc = acc_fn.calculate(validation_prediction, Y_test)

            wandb.log({"loss": loss, "val-loss": validation_loss, "epoch": epoch, "acc": acc})
            bar.set_description(
                f"Epoch: {epoch + 1}/{EPOCHS} batch: {batch_no} / "
                f"{batch_count}. Loss: {loss:.4f} Validation loss: {validation_loss:.4f}"
            )


pickle.dump(network, open(OUTPUT_DIR / f"encoded-classifier/{wandb_run.name}.pckl", "wb"))
wandb.finish()
