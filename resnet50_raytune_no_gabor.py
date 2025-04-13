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
from sklearn.metrics import precision_score, recall_score, f1_score
import tqdm
import cv2
import numpy as np
import ray
from ray import tune
from ray.tune import Trainable, CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Load and split the dataset
df = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\augmented_cleaned_converted_cancer.csv")
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create custom dataset class
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
        image = np.array(image)
        
        # Apply Gabor filter
        gabor_kernels = self.get_gabor_kernels()
        filtered_images = [cv2.filter2D(image, cv2.CV_8UC3, k) for k in gabor_kernels]
        # The output should be 4 channels (one for each Gabor kernel orientation)
        image = np.stack(filtered_images, axis=2)
        
        if self.transform:
            image = Image.fromarray(image.astype('uint8'), 'RGB')
            image = self.transform(image)
        
        label = self.class_to_idx[self.dataframe.iloc[idx, 1]]
        return image, torch.tensor(label)
    
    def get_gabor_kernels(self):
        ksize = 25  # Size of the filter
        sigma = 4.0  # Standard deviation of the Gaussian function
        theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Orientations of the Gabor filter
        lambd = 10.0  # Wavelength of the sinusoidal factor
        gamma = 0.5  # Spatial aspect ratio
        psi = 0  # Phase offset
        kernels = []
        for t in theta:
            kern = cv2.getGaborKernel((ksize, ksize), sigma, t, lambd, gamma, psi, ktype=cv2.CV_32F)
            kernels.append(np.array(kern, dtype=np.float32))  # Keep as numpy array
        return kernels


# Initialize datasets and dataloaders
train_dataset = CustomDataset(train_df, 'path_to_images', transform=transform)
val_dataset = CustomDataset(val_df, 'path_to_images', transform=transform)
test_dataset = CustomDataset(test_df, 'path_to_images', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Custom ResNet-50 model with Gabor filters
# Custom ResNet-50 model with Gabor filters
class CustomResNetWithGabor(nn.Module):
    def __init__(self):
        super(CustomResNetWithGabor, self).__init__()
        self.gabor_kernels = self.get_gabor_kernels()
        self.resnet = models.resnet50(weights=None)
        # Replace the first convolutional layer with Gabor filter application
        self.resnet.conv1 = nn.Conv2d(len(self.gabor_kernels), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(self.resnet.fc.in_features, 7)  # Replace num_classes with your number of classes
        )

    def get_gabor_kernels(self):
        ksize = 25  # Size of the filter
        sigma = 4.0  # Standard deviation of the Gaussian function
        theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Orientations of the Gabor filter
        lambd = 10.0  # Wavelength of the sinusoidal factor
        gamma = 0.5  # Spatial aspect ratio
        psi = 0  # Phase offset
        kernels = []
        for t in theta:
            kern = cv2.getGaborKernel((ksize, ksize), sigma, t, lambd, gamma, psi, ktype=cv2.CV_32F)
            kernels.append(kern)  # Keep as numpy array
        return kernels

    def forward(self, x):
        x = self.resnet(x)
        return x


# Training function for Ray Tune
def train_resnet(config):
    # Initialize CSV file for saving metrics
    csv_file = './training_metrics_ray_tune.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1_score'])

    model = CustomResNetWithGabor().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(50):  # Training for 50 epochs
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        val_precision = precision_score(all_labels, all_predictions, average='weighted')
        val_recall = recall_score(all_labels, all_predictions, average='weighted')
        val_f1 = f1_score(all_labels, all_predictions, average='weighted')

        # Save metrics to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, epoch_loss, val_loss, val_acc, val_precision, val_recall, val_f1])

        # Send the current metrics to Ray Tune
        tune.report(loss=val_loss, accuracy=val_acc)

# Define the search space
config = {
    "lr": tune.grid_search([0.001, 0.0001])
}

# Set up the scheduler
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=10,
    grace_period=1,
    reduction_factor=2)

# Run the Ray Tune experiment
analysis = tune.run(
    train_resnet,
    resources_per_trial={"cpu": 4, "gpu": 1},
    config=config,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"]))

print("Best hyperparameters found were: ", analysis.best_config)

best_trial = analysis.best_trial 
best_checkpoint_dir = best_trial.checkpoint.value 
model_state_dict = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
model = CustomResNetWithGabor() 
model.load_state(model.state_dict() , "best_withgabor_resnet50_model.pth")