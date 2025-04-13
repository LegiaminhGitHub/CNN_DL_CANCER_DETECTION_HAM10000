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
import cv2
import numpy as np

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
        image = np.array(image)

        # Apply Gabor filter
        gabor_kernels = self.get_gabor_kernels()
        filtered_images = [cv2.filter2D(image, cv2.CV_8UC3, k) for k in gabor_kernels]
        image = np.concatenate(filtered_images, axis=-1)

        if self.transform:
            image = Image.fromarray(image.astype('uint8'), 'RGB')
            image = self.transform(image)

        label = self.class_to_idx[self.dataframe.iloc[idx, 1]]
        return image, torch.tensor(label)

    def get_gabor_kernels(self):
        # Define Gabor filter parameters
        ksize = 25  # Size of the filter
        sigma = 4.0  # Standard deviation of the Gaussian function
        theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Orientations of the Gabor filter
        lambd = 10.0  # Wavelength of the sinusoidal factor
        gamma = 0.5  # Spatial aspect ratio
        psi = 0  # Phase offset

        kernels = []
        for t in theta:
            kern = cv2.getGaborKernel((ksize, ksize), sigma, t, lambd, gamma, psi, ktype=cv2.CV_32F)
            kernels.append(kern)

        return kernels

# Initialize datasets and dataloaders
train_dataset = CustomDataset(train_df, 'path_to_images', transform=transform)
val_dataset = CustomDataset(val_df, 'path_to_images', transform=transform)
test_dataset = CustomDataset(test_df, 'path_to_images', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize DenseNet-161 model with dropout
class CustomDenseNet(nn.Module):
    def __init__(self):
        super(CustomDenseNet, self).__init__()
        self.model = models.densenet161(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout layer
            nn.Linear(self.model.classifier.in_features, 7)  # Replace num_classes with your number of classes
        )

    def forward(self, x):
        return self.model(x)

model = CustomDenseNet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define early stopping criteria
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=5, min_delta=0.01)

# Prepare to record metrics
csv_file = './training_metrics_Densenet_with_gabor_latest_v3.csv'
fields = ['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1_score', 'val_auc']
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(fields)

# Training and validation loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}', flush=True)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(val_loader, desc=f"Validation {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total
    val_precision = precision_score(all_labels, all_predictions, average='weighted')
    val_recall = recall_score(all_labels, all_predictions, average='weighted')
    val_f1 = f1_score(all_labels, all_predictions, average='weighted')
    # Convert labels and predictions to one-hot format for AUC calculation
    # all_labels_binarized = label_binarize(all_labels, classes=[0, 1, 2, 3, 4, 5, 6])

    # Record metrics in CSV
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, epoch_loss, val_loss, val_acc, val_precision, val_recall, val_f1])
    
    # Check early stopping condition
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Save the model checkpoint
model_save_path = './densenet_model_gabor_filter.pth'
torch.save(model.state_dict(), model_save_path)

# To load the model later, use this
model = CustomDenseNet()
model.load_state_dict(torch.load(model_save_path))
model = model.to(device)