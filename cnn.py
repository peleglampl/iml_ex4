import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from xgboost import XGBClassifier


class ResNet18(nn.Module):
    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])            
            self.resnet18 = resnet18()
                
        
        in_features_dim = self.resnet18.fc.in_features        
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                    param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        logits = self.logistic_regression(features)  # Binary classification- output single logit
        return logits


def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(imgs)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    return acc

def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader)):
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader)

# Set the random seed for reproducibility
torch.manual_seed(0)

### UNCOMMENT THE FOLLOWING LINES TO TRAIN THE MODEL ###
# From Scratch
model = ResNet18(pretrained=False, probing=False)
# Linear probing
model = ResNet18(pretrained=True, probing=True)
# Fine-tuning
model = ResNet18(pretrained=True, probing=False)

transform = model.transform
batch_size = 32  # batch size of 32
num_of_epochs = 50
learning_rate = 0.0001
path = "C:\Users\peleg\IML\whichfaceisreal" # For example '/cs/usr/username/whichfaceisreal/'
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
### Define the loss function and the optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
### Train the model

# Train the model
for epoch in range(num_of_epochs):
    # Run a training epoch
    loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
    # Compute the accuracy
    train_acc = compute_accuracy(model, train_loader, device)
    # Compute the validation accuracy
    val_acc = compute_accuracy(model, val_loader, device)
    print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
    # Stopping condition
    ### YOUR CODE HERE ###
    if val_acc >= 0.99:
        print("Early stopping as validation accuracy reached 99%")
        break

# Compute the test accuracy
test_acc = compute_accuracy(model, test_loader, device)

### Bonus part: Linear Probing based on sklearn
def linear_probing(train_loader, test_loader, device):
    # Extract features using the pretrained ResNet18
    model = ResNet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()  # Remove the final classification layer
    model = model.to(device)
    model.eval()

    # extract the features
    def extract_features(data_loader):
        features = []
        labels = []
        with torch.no_grad():
            for imgs, lbls in data_loader:
                imgs = imgs.to(device)
                feats = model(imgs)
                features.append(feats.cpu().numpy())
                labels.append(lbls.cpu().numpy())
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, labels

    X_train, y_train = extract_features(train_loader)
    X_test, y_test = extract_features(test_loader)

    clf = self.logistic_regression = nn.Linear(in_features_dim, 1)
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f'Linear Probing Test Accuracy: {acc:.4f}')



def fine_tuning(train_loader, val_loader, test_loader, device):
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

    for lr in learning_rates:
        print("fine tuning with learning rate: ", lr)

        model = ResNet18(pretrained=True, probing=False)
        model = model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer

        # train one epoch
        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)

        # evaluate accuracy
        val_acc = compute_accuracy(model, val_loader, device)
        train_acc = compute_accuracy(model, train_loader, device)
        test_acc = compute_accuracy(model, test_loader, device)

        print(f"LR={lr} | Loss={loss:.4f} | "
              f"Train Acc={train_acc:.4f} | "
              f"Val Acc={val_acc:.4f} | "
              f"Test Acc={test_acc:.4f}")


