import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import tqdm
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session

# Load and split the dataset
df = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\augmented_cleaned_converted_cancer.csv")
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print(f'Train set size: {len(train_df)}')
print(f'Validation set size: {len(val_df)}')
print(f'Test set size: {len(test_df)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Image Path'])
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[self.dataframe.iloc[idx, 1]]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

# Initialize datasets and dataloaders
train_dataset = CustomDataset(train_df, 'path_to_images', transform=transform)
val_dataset = CustomDataset(val_df, 'path_to_images', transform=transform)
test_dataset = CustomDataset(test_df, 'path_to_images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize Hybrid ResNet50-DenseNet model with dropout
class HybridResNetDenseNet(nn.Module):
    def __init__(self, config):
        super(HybridResNetDenseNet, self).__init__()
        self.resnet = models.resnet50(weights=None)
        self.densenet = models.densenet161(weights=None)
        num_features_resnet = self.resnet.fc.in_features
        num_features_densenet = self.densenet.classifier.in_features
        self.resnet.fc = nn.Identity()
        self.densenet.classifier = nn.Identity()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features_resnet + num_features_densenet, 7)
        )

    def forward(self, x):
        resnet_features = self.resnet(x)
        densenet_features = self.densenet(x)
        combined_features = torch.cat((resnet_features, densenet_features), dim=1)
        output = self.fc(combined_features)
        return output

def train_model(config, checkpoint_dir=None):
    model = HybridResNetDenseNet(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    csv_file = 'training_metrics_Hybrid_ResNet_DenseNet_no_gabor_latest_v5.csv'
    fields = ['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1_score', 'val_auc']

    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    for epoch in range(30):  # Simplified for Ray Tune
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{50}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{50}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_acc = correct / total
        val_precision = precision_score(all_labels, all_predictions, average='weighted')
        val_recall = recall_score(all_labels, all_predictions, average='weighted')
        val_f1 = f1_score(all_labels, all_predictions, average='weighted')

        # Record metrics in CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_loss, val_loss, val_acc, val_precision, val_recall, val_f1])

        # Report metrics and checkpoint to Ray Tune
        session.report({
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        # Save checkpoint
        if checkpoint_dir:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch
            }, os.path.join(checkpoint_dir, "checkpoint"))

    # Final report to ensure the last epoch is captured
    session.report({
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

    # Final save for the model state
    if checkpoint_dir:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch
        }, os.path.join(checkpoint_dir, "final_checkpoint"))






# Main function to execute the tuning and save the best model
def main(num_samples=10, max_num_epochs=50, gpus_per_trial=1):
    config = {
        "lr": tune.loguniform(1e-4, 1e-2)
    }
    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        metric_columns=["val_loss", "val_accuracy", "training_iteration"]
    )
    result = tune.run(
    tune.with_parameters(train_model),
    resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    progress_reporter=reporter,
)

    best_trial = result.get_best_trial("val_accuracy", "max", "last")
    best_trained_model = HybridResNetDenseNet(config)
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = 0
    best_trained_model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = best_trained_model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_acc += (predicted == labels).sum().item()
    test_acc /= len(test_loader.dataset)
    print("Best trial test set accuracy: {}".format(test_acc))
    model_save_path = 'hybrid_resnet_densenet_best_model_trained.pth'
    torch.save(best_trained_model.state_dict(), model_save_path)
if __name__ == "__main__":
    main()
