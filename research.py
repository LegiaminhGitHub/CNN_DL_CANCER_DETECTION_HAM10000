import torch
import torch.nn as nn  # Make sure this is imported
from torchvision import models, transforms
from PIL import Image

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
class CustomDenseNet(nn.Module):
    def __init__(self):
        super(CustomDenseNet, self).__init__()
        self.model = models.densenet161(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.model.classifier.in_features, 7)  # Adjust the number of classes as needed
        )

    def forward(self, x):
        return self.model(x)

model = CustomDenseNet().to(device)
model.load_state_dict(torch.load('densenet_model.pth'))
model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to make a prediction
def predict(image_path, model, transform, class_to_idx):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return idx_to_class[predicted.item()]

# Example usage
image_path = "C:\\Users\\ADMIN\\Downloads\\Actinic keratosis.jpg"  # Replace with your image path
class_to_idx = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}  # Your class labels

prediction = predict(image_path, model, transform, class_to_idx)
print(f'Predicted class: {prediction}')
