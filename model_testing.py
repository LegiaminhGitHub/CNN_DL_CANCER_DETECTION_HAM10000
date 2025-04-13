import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms

# Define the number of classes
num_classes = 6  # Update this based on your specific use case

# Initialize the model with the correct architecture
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Load the state dictionary
state_dict = torch.load('C:\\Users\\ADMIN\\Desktop\\cancer_research_final\\densenet_model_testing_v5.pth', map_location=torch.device('cpu'))  # Update map_location if using GPU
model.load_state_dict(state_dict)

# Switch the model to evaluation mode
model.eval()

# Define the transformations for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
image_path = r"C:\Users\ADMIN\Downloads\angioma_cherry_high.jpg"
image = Image.open(image_path).convert('RGB')
image = transform(image)
image = image.unsqueeze(0)  # Add batch dimension

# Perform prediction
output = model(image)

# Interpret the results
_, predicted = torch.max(output , 1)
print('Predicted class:', predicted.item())
