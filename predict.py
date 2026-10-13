import sys
import os
import torch
from PIL import Image
from torch import nn, load
from torchvision import transforms

class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64 * (28 - 6) * (28 - 6), 10)  
        )

    def forward(self, x): 
        return self.model(x)

def predict(image_path, model_path='model_state.pt'):
    # Load model architecture and weights
    clf = ImageClassifier().to('cpu')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model with train.py first.")

    with open(model_path, 'rb') as f:
        clf.load_state_dict(load(f, map_location='cpu'))
    clf.eval()

    # Preprocessing matching MNIST normalization
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to('cpu')

    with torch.no_grad():
        output = clf(img_tensor)
        predicted_digit = torch.argmax(output, dim=1).item()
        probabilities = torch.softmax(output, dim=1)[0]
        confidence = probabilities[predicted_digit].item()

    print(f"[*] Input Image     : {image_path}")
    print(f"[*] Predicted Digit : {predicted_digit}")
    print(f"[*] Confidence      : {confidence * 100:.2f}%")
    return predicted_digit

if __name__ == "__main__":
    target_img = sys.argv[1] if len(sys.argv) > 1 else 'img_3.jpg'
    predict(target_img)