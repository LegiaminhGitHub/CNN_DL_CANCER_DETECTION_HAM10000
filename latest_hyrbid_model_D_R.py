import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import ray
from ray import tune
from ray.tune import CLIReporter
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import tqdm
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load and split the dataset
df = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\augmented_cleaned_converted_cancer.csv")
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Define a custom dataset
class CancerDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        # Update 'Label' to the actual column name
        self.class_to_idx = {label: idx for idx, label in enumerate(dataframe['Class'].unique())}  # Adjust as needed

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Image Path'])
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[self.dataframe.iloc[idx]['Class']]  # Adjust as needed
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)


# Define the hybrid model
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        self.densenet = models.densenet121(weights='DEFAULT')
        self.resnet = models.resnet50(pretrained=True)

        # Modify the classifier layers
        self.densenet.classifier = nn.Linear(self.densenet.classifier.in_features, num_classes)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # Combine the outputs
        self.fc = nn.Linear(num_classes * 2, num_classes)  # Adjust for combined output

    def forward(self, x):
        densenet_out = self.densenet(x)
        resnet_out = self.resnet(x)
        combined_out = torch.cat((densenet_out, resnet_out), dim=1)
        return self.fc(combined_out)

# Define the training function with early stopping and metric collection
def train_model(config):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create datasets and dataloaders
    train_dataset = CancerDataset(train_df,"path_to_images" ,  transform=transform)
    val_dataset = CancerDataset(val_df,"path_to_images", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize the hybrid model
    model = HybridModel(num_classes=len(train_df.iloc[:, 1].unique()))
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Early stopping parameters
    patience = config["patience"]
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = "best_hybrid_resnet_densenet_model.pth"  # Path to save the best model

    # Metrics storage
    metrics = []

    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        for images, labels in tqdm.tqdm(train_loader):
            print('##############################################################################' + str(type(images)) + ', ' + str(type(labels)) + '###################################')  # Check the types  
            images = images.to(device)
            labels = labels.to(device)

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / len(val_dataset)
        val_loss /= len(val_loader)

        # Store metrics
        metrics.append({
            'epoch': epoch + 1,
            'train_loss': loss.item(),
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        })

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            # print(f"Best model saved at epoch {epoch + 1} with val_loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            # print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        tune.report(val_loss=val_loss, val_accuracy=val_accuracy)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('training_metrics_Hybrid_ResNet_DenseNet_no_gabor_latest_v5.csv', index=False)

# Define the search space for hyperparameters
config = {
    "batch_size": tune.choice([16, 32]),
    "lr": tune.loguniform(1e-5, 1e-1),
    "epochs": tune.choice([30, 100]),
    "patience": tune.choice([6, 5])  # Number of epochs to wait for improvement
}

# Initialize Ray
ray.init()

# Run the tuning process
analysis = tune.run(
    train_model,
    config=config,
    num_samples=10,
    resources_per_trial={"cpu": 1, "gpu": 1},
    progress_reporter=CLIReporter(metric_columns=["val_loss", "val_accuracy"])
)

# Print the best configuration
best_config = analysis.get_best_config(metric="val_accuracy", mode="max")
print("Best config: ", best_config)
