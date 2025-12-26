import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from helpers import *
import pandas as pd
import torch.optim as optim


class EuropeDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
        """
        # Load the data into a tensors
        data = pd.read_csv(csv_file)
        # The features shape is (n,d)
        # The labels shape is (n)
        # The feature dtype is float
        # THe labels dtype is long
        self.features = torch.tensor(data[['long', 'lat']].values, dtype=torch.float32)
        # labels: (n, )
        self.labels = torch.tensor(data['country'].values, dtype=torch.long)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data row
        
        Returns:
            dictionary or list corresponding to a feature tensor and it's corresponding label tensor
        """
        return self.features[idx], self.labels[idx]


class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim):
        super(MLP, self).__init__()
        """
        Args:
            num_hidden_layers (int): The number of hidden layers, in total you'll have an extra layer at the end, from hidden_dim to output_dim
            hidden_dim (int): The hidden layer dimension
            output_dim (int): The output dimension, should match the number of classes in the dataset
        """
        layers = []
        layers.append(nn.Linear(2, hidden_dim))  # input layer
        layers.append(nn.ReLU())

        # the hidden layers:
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())

        # the output layer:
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)  # chains the layers together and runs them one after the other

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)


########## PART 6.1: TRAINING THE MLP MODEL ##########

def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256):
    # getting the data loaders in batches:
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

    # initialize your criterion and optimizer here
    criterion = nn.CrossEntropyLoss()  # loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)  # optimizer

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    train_batch_losses = []
    tracked_layers = [0, 30, 60, 90, 95, 99]  # layers to monitor
    linear_layers = [layer for layer in model.network[:-1] if isinstance(layer, nn.Linear)]
    epoch_sums = {layer: 0.0 for layer in tracked_layers}
    epoch_means = {layer: [] for layer in tracked_layers}

    for ep in range(epochs):
        model.train()  # set model to training mode
        # perform training epoch here
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_samples = 0
        # iterate over batches and train:
        for batch in trainloader:
            x, y = batch
            predictions = model.forward(x)  # forward pass
            loss = criterion(predictions, y)  # the error
            optimizer.zero_grad()  # zero the gradients
            # compute gradients and update weights
            loss.backward()

            # 6.2.5: adding monitoring gradients:
            for layer_idx in tracked_layers:
                W = linear_layers[layer_idx].weight
                if W.grad is not None:
                    grad_mag = (W.grad ** 2).sum().item()  # ||grad||_2^2
                    epoch_sums[layer_idx] += grad_mag

            optimizer.step()  # update weights
            train_batch_losses.append(loss.item())
            epoch_train_loss += loss.item() * x.size(0)
            _, predicted = torch.max(predictions, 1)
            epoch_train_correct += (predicted == y).sum().item()
            epoch_train_samples += x.size(0)

        num_batches = len(trainloader)

        for layer_idx in tracked_layers:
            mean_grad = epoch_sums[layer_idx] / num_batches
            epoch_means[layer_idx].append(mean_grad)
            epoch_sums[layer_idx] = 0.0

        # compute epoch loss and accuracy
        train_loss = epoch_train_loss / epoch_train_samples
        train_acc = epoch_train_correct / epoch_train_samples
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            # perform validation loop and test loop here
            epoch_val_loss = 0.0
            epoch_val_correct = 0
            epoch_val_samples = 0
            epoch_test_loss = 0.0
            epoch_test_correct = 0
            epoch_test_samples = 0
            # predict, loss, accuracy for val and test sets:
            # how is the model doing on the validation set?
            for batch in valloader:
                x, y = batch
                predictions = model.forward(x)
                loss = criterion(predictions, y)
                epoch_val_loss += loss.item() * x.size(0)
                _, predicted = torch.max(predictions, 1)
                epoch_val_correct += (predicted == y).sum().item()
                epoch_val_samples += x.size(0)
            # how is the model doing on the test set?
            for batch in testloader:
                x, y = batch
                predictions = model.forward(x)
                loss = criterion(predictions, y)
                epoch_test_loss += loss.item() * x.size(0)
                _, predicted = torch.max(predictions, 1)
                epoch_test_correct += (predicted == y).sum().item()
                epoch_test_samples += x.size(0)

            val_loss = epoch_val_loss / epoch_val_samples
            val_acc = epoch_val_correct / epoch_val_samples
            test_loss = epoch_test_loss / epoch_test_samples
            test_acc = epoch_test_correct / epoch_test_samples
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1],
                                                                                       test_accs[-1]))

    return (model, train_accs, val_accs, test_accs,
            train_losses, val_losses, test_losses,
            train_batch_losses, epoch_means)


def learning_rate_experiment(train_dataset, val_dataset, test_dataset):
    val_losses_per_lr = {}
    # PART 1: MLP model for diff. Learning Rates:
    # Find the number of classes, e.g.:
    output_dim = len(train_dataset.labels.unique())
    learning_rates = [1, 0.01, 0.001, 0.00001]
    for lr in learning_rates:
        # Create the model
        model = MLP(6, 16, output_dim)
        # train the model
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, train_batch_losses = train(
            train_dataset,
            val_dataset, test_dataset,
            model, lr=lr,
            epochs=50, batch_size=256)
        val_losses_per_lr[lr] = val_losses  # store the val losses for this learning rate
    # PLOT:
    plt.figure()
    for lr, losses in val_losses_per_lr.items():
        plt.plot(losses, label=f'LR={lr}')

    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Epochs for Different Learning Rates")
    plt.legend()
    plt.show()


def epochs_experiment(train_dataset, val_dataset, test_dataset):
    # PART 2: MLP model for diff. Epochs:
    output_dim = len(train_dataset.labels.unique())
    model = MLP(6, 16, output_dim)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, train_batch_losses = train(
        train_dataset,
        val_dataset, test_dataset,
        model, lr=0.001,
        epochs=100, batch_size=256)
    # PLOT:
    plt.figure()
    plt.plot(train_losses, label='Train', color='red')
    plt.plot(val_losses, label='Val', color='blue')
    plt.plot(test_losses, label='Test', color='green')
    plt.title('Losses over 100 Epochs')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train', color='red')
    plt.plot(val_accs, label='Val', color='blue')
    plt.plot(test_accs, label='Test', color='green')
    plt.title('Accs. over 100 Epochs')
    plt.legend()
    plt.show()


def batch_size_experiment(train_dataset, val_dataset, test_dataset):
    batch_sizes = [1, 16, 128, 1024]
    epochs_sizes = [1, 10, 50, 50]
    validation_acc_pre_bs = {}
    training_loss_per_bs = {}
    output_dim = len(train_dataset.labels.unique())

    for batch_size, epochs in zip(batch_sizes, epochs_sizes):
        model = MLP(6, 16, output_dim)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, train_batch_losses = train(
            train_dataset,
            val_dataset, test_dataset,
            model, lr=0.001,
            epochs=epochs,
            batch_size=batch_size)
        # (ii) Speed: iterations per epoch
        iterations = len(train_dataset) // batch_size  # number of iterations per epoch
        print('Batch Size {:}, Epochs {:}, Iterations per Epoch {:}'.format(batch_size, epochs, iterations))
        # store results:
        validation_acc_pre_bs[batch_size] = val_accs
        training_loss_per_bs[batch_size] = train_batch_losses

    # (i) plotting accuracy vs epochs for different batch sizes:
    plt.figure()
    for batch_size, val_accs in validation_acc_pre_bs.items():
        plt.plot(val_accs, label=f'Batch Size={batch_size}')
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Epochs for Different Batch Sizes")
    plt.legend()
    plt.show()

    # (iii) calculation stability: training loss vs batch
    plt.figure()
    for batch_size, batch_losses in training_loss_per_bs.items():
        plt.plot(batch_losses, label=f'Batch Size={batch_size}')
    plt.xlabel("Batch index")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Batch for Different Batch Sizes")
    plt.legend()
    plt.show()

#
# ######## PART 6.2: EVALUATING MLPs PERFORMANCE ##########
# def train_several_mlp(train_dataset, val_dataset, test_dataset):
#     depth_arr = [1, 2, 6, 10, 6, 6, 6]
#     width_arr = [16, 16, 16, 16, 8, 32, 64]
#     models = []
#     results = []
#     for depth, width in zip(depth_arr, width_arr):
#         output_dim = len(train_dataset.labels.unique())
#         model = MLP(depth, width, output_dim)
#         model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _ = train(
#             train_dataset, val_dataset, test_dataset,
#             model, lr=0.001, epochs=50, batch_size=256
#         )
#         # store the model and results
#         models.append(model)
#         results.append({
#             "depth": depth,
#             "width": width,
#             "final_val_acc": val_accs[-1],
#             "final_test_acc": test_accs[-1],
#             "train_accs": train_accs,
#             "train_losses": train_losses,
#             "val_losses": val_losses,
#             "test_losses": test_losses
#         })
#
#         print(f"Depth {depth}, Width {width}, "
#               f"Train {train_accs[-1]:.3f}, "
#               f"Val {val_accs[-1]:.3f}, "
#               f"Test {test_accs[-1]:.3f}")
#     # best model selection based on validation accuracy
#     best_idx = max(range(len(results)), key=lambda i: results[i]["final_val_acc"])
#     best_model = models[best_idx]
#     best_result = results[best_idx]
#
#     print("Best model:",
#           "Depth =", best_result["depth"],
#           "Width =", best_result["width"])
#
#     # plot the losses of the best model
#     plt.figure()
#     plt.plot(best_result["train_losses"], label="Train")
#     plt.plot(best_result["val_losses"], label="Val")
#     plt.plot(best_result["test_losses"], label="Test")
#     plt.title(f"Best Model Losses (Depth={best_result['depth']}, Width={best_result['width']})")
#     plt.legend()
#     plt.show()
#
#     # plot the decision boundaries of the best model
#     plot_decision_boundaries(best_model, test_dataset.features.numpy(), test_dataset.labels.numpy(),
#                              title='Best MLP Model Decision Boundaries', implicit_repr=False)
#
#
#     # worst model selection based on validation accuracy
#     worst_idx = min(range(len(results)), key=lambda i: results[i]["final_val_acc"])
#     worst_model = models[worst_idx]
#     worst_result = results[worst_idx]
#     print("Worst model:",
#           "Depth =", worst_result["depth"],
#           "Width =", worst_result["width"])
#
#     # plot the losses of the worst model
#     plt.figure()
#     plt.plot(worst_result["train_losses"], label="Train")
#     plt.plot(worst_result["val_losses"], label="Val")
#     plt.plot(worst_result["test_losses"], label="Test")
#     plt.title(f"Worst Model Losses (Depth={worst_result['depth']}, Width={worst_result['width']})")
#     plt.legend()
#     plt.show()
#
#     # plot the decision boundaries of the worst model
#     plot_decision_boundaries(worst_model, test_dataset.features.numpy(), test_dataset.labels.numpy(),
#                              title='Worst MLP Model Decision Boundaries', implicit_repr=False)
#
#
#     # Depth of Network: 6.2.3: Accuracy vs Depth for width = 16
#     width_16_results = [res for res in results if res["width"] == 16]
#     depths = [res["depth"] for res in width_16_results]
#     train_acc_depth = [res["train_accs"][-1] for res in width_16_results]
#     val_accs_depth = [res["final_val_acc"] for res in width_16_results]
#     test_accs_depth = [res["final_test_acc"] for res in width_16_results]
#
#     # plot accuracy vs depth for width = 16
#     plt.figure()
#     plt.plot(depths, train_acc_depth, marker='o', label='Train')
#     plt.plot(depths, val_accs_depth, marker='o', label='Validation')
#     plt.plot(depths, test_accs_depth, marker='o', label='Test')
#     plt.xlabel("Number of Hidden Layers")
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy vs Network Depth (Width = 16)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
#     # Width of Network: 6.2.4: Accuracy vs Width for depth = 6
#     depth_6_results = [res for res in results if res["depth"] == 6]
#     widths = [res["width"] for res in depth_6_results]  # number of neurons in hidden layers
#     training_acc_width = [res["train_accs"][-1] for res in depth_6_results]
#     validation_accs_width = [res["final_val_acc"] for res in depth_6_results]
#     testing_accs_width = [res["final_test_acc"] for res in depth_6_results]
#
#     # plot accuracy vs width for depth = 6
#     plt.figure()
#     plt.plot(widths, training_acc_width, marker='o', label='Train')
#     plt.plot(widths, validation_accs_width, marker='o', label='Validation')
#     plt.plot(widths, testing_accs_width, marker='o', label='Test')
#     plt.xlabel("Number of Neurons in Hidden Layers")
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy vs Network Depth (Width = 16)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


def train_models(train_dataset, val_dataset, test_dataset, depth_arr, width_arr,
                 lr=0.001, epochs=50, batch_size=256):

    models = []
    results = []
    output_dim = len(train_dataset.labels.unique())

    for depth, width in zip(depth_arr, width_arr):
        model = MLP(depth, width, output_dim)

        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, _ = train(
            train_dataset, val_dataset, test_dataset,
            model, lr=lr, epochs=epochs, batch_size=batch_size
        )

        models.append(model)
        results.append({
            "depth": depth,
            "width": width,
            "final_val_acc": val_accs[-1],
            "final_test_acc": test_accs[-1],
            "train_accs": train_accs,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_losses": test_losses
        })

        print(f"Depth {depth}, Width {width}, "
              f"Train {train_accs[-1]:.3f}, "
              f"Val {val_accs[-1]:.3f}, "
              f"Test {test_accs[-1]:.3f}")

    return models, results


def select_best_and_worst(models, results):
    best_idx = max(range(len(results)), key=lambda i: results[i]["final_val_acc"])
    worst_idx = min(range(len(results)), key=lambda i: results[i]["final_val_acc"])

    return (
        models[best_idx], results[best_idx],
        models[worst_idx], results[worst_idx]
    )


def plot_model_losses(result, title_prefix):
    plt.figure()
    plt.plot(result["train_losses"], label="Train")
    plt.plot(result["val_losses"], label="Val")
    plt.plot(result["test_losses"], label="Test")
    plt.title(f"{title_prefix} (Depth={result['depth']}, Width={result['width']})")
    plt.legend()
    plt.show()


def plot_accuracy_vs_depth(results, fixed_width):
    filtered = [r for r in results if r["width"] == fixed_width]

    depths = [r["depth"] for r in filtered]
    train_acc = [r["train_accs"][-1] for r in filtered]
    val_acc = [r["final_val_acc"] for r in filtered]
    test_acc = [r["final_test_acc"] for r in filtered]

    plt.figure()
    plt.plot(depths, train_acc, marker='o', label='Train')
    plt.plot(depths, val_acc, marker='o', label='Validation')
    plt.plot(depths, test_acc, marker='o', label='Test')
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Depth (Width = {fixed_width})")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy_vs_width(results, fixed_depth):
    filtered = [r for r in results if r["depth"] == fixed_depth]

    widths = [r["width"] for r in filtered]
    train_acc = [r["train_accs"][-1] for r in filtered]
    val_acc = [r["final_val_acc"] for r in filtered]
    test_acc = [r["final_test_acc"] for r in filtered]

    plt.figure()
    plt.plot(widths, train_acc, marker='o', label='Train')
    plt.plot(widths, val_acc, marker='o', label='Validation')
    plt.plot(widths, test_acc, marker='o', label='Test')
    plt.xlabel("Number of Neurons in Hidden Layers")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Width (Depth = {fixed_depth})")
    plt.legend()
    plt.grid(True)
    plt.show()


def train_several_mlp(train_dataset, val_dataset, test_dataset):
    depth_arr = [1, 2, 6, 10, 6, 6, 6]
    width_arr = [16, 16, 16, 16, 8, 32, 64]

    models, results = train_models(
        train_dataset, val_dataset, test_dataset,
        depth_arr, width_arr
    )

    best_model, best_result, worst_model, worst_result = select_best_and_worst(models, results)

    print("Best model:", best_result["depth"], best_result["width"])
    plot_model_losses(best_result, "Best Model Losses")
    plot_decision_boundaries(best_model,
                             test_dataset.features.numpy(),
                             test_dataset.labels.numpy(),
                             title="Best Model Decision Boundaries",
                             implicit_repr=False)

    print("Worst model:", worst_result["depth"], worst_result["width"])
    plot_model_losses(worst_result, "Worst Model Losses")
    plot_decision_boundaries(worst_model,
                             test_dataset.features.numpy(),
                             test_dataset.labels.numpy(),
                             title="Worst Model Decision Boundaries",
                             implicit_repr=False)

    plot_accuracy_vs_depth(results, fixed_width=16)
    plot_accuracy_vs_width(results, fixed_depth=6)

# Monitoring Gradients:
# 100 hidden layers, 4 neurons each, lr=0.001, batch size=256, epochs=10


# grad_magnitudes = ||  grad ||
def monitor_gradients(train_dataset, val_dataset, test_dataset):
    depth = 100
    width = 4
    output_dim = len(train_dataset.labels.unique())
    model = MLP(depth, width, output_dim)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, train_batch_losses, epoch_means = train(
        train_dataset,
        val_dataset, test_dataset,
        model, lr=0.001,
        epochs=10, batch_size=256)

    for layer_idx, values in epoch_means.items():
        print(layer_idx, values)

    # Plotting the gradient magnitudes
    plt.figure()
    for layer_idx, values in epoch_means.items():
        plt.plot(values, label=f"Layer {layer_idx}")

    plt.xlabel("Epoch")
    plt.ylabel("Mean Gradient Magnitude")
    plt.title("Gradient Magnitude vs Epoch")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(0)

    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    # Calling the experiments:
    # learning_rate_experiment(train_dataset, val_dataset, test_dataset)
    # epochs_experiment(train_dataset, val_dataset, test_dataset)
    # batch_size_experiment(train_dataset, val_dataset, test_dataset)
    # train_several_mlp(train_dataset, val_dataset, test_dataset)
    monitor_gradients(train_dataset, val_dataset, test_dataset)


    # # Example of training a single MLP model
    # plt.plot(train_losses, label='Train', color='red')
    # plt.plot(val_losses, label='Val', color='blue')
    # plt.plot(test_losses, label='Test', color='green')
    # plt.title('Losses')
    # plt.legend()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(train_accs, label='Train', color='red')
    # plt.plot(val_accs, label='Val', color='blue')
    # plt.plot(test_accs, label='Test', color='green')
    # plt.title('Accs.')
    # plt.legend()
    # plt.show()

    #
    # train_data = pd.read_csv('train.csv')
    # val_data = pd.read_csv('validation.csv')
    # test_data = pd.read_csv('test.csv')
    # plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=False)
