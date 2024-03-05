import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import os
from tqdm import tqdm
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define a dataset class to load images and questions
class VQADataset(Dataset):
    def __init__(self, data_dir, json_file, tokenizer, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.tokenizer = tokenizer

        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.data_dir, item['image_id'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        question = item['question']
        tokenized_question = self.tokenizer(question, return_tensors='pt', padding=True, truncation=True)
        
        return image, tokenized_question, item['qa_pairs'][0]['answer']

# Define a fusion module to combine image and text features
class FusionModule(nn.Module):
    def __init__(self, input_dim_image, input_dim_text, output_dim):
        super(FusionModule, self).__init__()
        self.fc_image = nn.Linear(input_dim_image, output_dim)
        self.fc_text = nn.Linear(input_dim_text, output_dim)

    def forward(self, image_features, text_features):
        fused_features = torch.cat((self.fc_image(image_features), self.fc_text(text_features)), dim=1)
        return fused_features

# Define your VQA model
class VQA(nn.Module):
    def __init__(self, resnet, bert_model, fusion_module, num_classes):
        super(VQA, self).__init__()
        self.resnet = resnet
        self.bert = bert_model
        self.fusion = fusion_module
        self.fc = nn.Linear(fusion_dim, num_classes)

    def forward(self, image, question):
        # Extract image features
        image_features = self.resnet(image)
        # Tokenize and process question
        outputs = self.bert(**question)
        text_features = outputs.last_hidden_state.mean(dim=1)  # Pooling strategy (mean pooling)
        # Combine features
        fused_features = self.fusion(image_features, text_features)
        # Final classification
        output = self.fc(fused_features)
        return output

# Load pre-trained ResNet-18 model from .pth file
def load_resnet18(path):
    resnet = models.resnet18()
    resnet.fc = nn.Identity()  # Remove final fully connected layer
    resnet.load_state_dict(torch.load(path))
    resnet.eval()
    return resnet

# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained ResNet-18 model
resnet_model_path = "path/to/resnet18.pth"
resnet = load_resnet18(resnet_model_path)

# Load pre-trained BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Instantiate Fusion Module
resnet_output_dim = 512  # Output dimension of ResNet-18
fusion_dim = 768  # Dimensionality of BERT hidden states
fusion_module = FusionModule(resnet_output_dim, fusion_dim, fusion_dim)

# Instantiate VQA model
vqa_model = VQA(resnet, bert_model, fusion_module, num_classes=2)  # Adjust num_classes as needed

# Define evaluation function
def evaluate_vqa_model(vqa_model, data_loader, device):
    vqa_model.eval()
    with torch.no_grad():
        for images, questions, answers in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            questions = {key: val.to(device) for key, val in questions.items()}
            
            # Forward pass
            outputs = vqa_model(images, questions)
            
            # Compute attention weights (assuming self-attention in BERT)
            attention_weights = vqa_model.bert.encoder.layer[-1].attention.self.attention_probs
            attention_weights = attention_weights.mean(dim=1)  # Average attention weights across heads
            attention_weights = attention_weights.squeeze(0)  # Remove batch dimension
            
            # Visualize attention weights for the image
            visualize_attention(images[0], attention_weights)

# Define a function to visualize attention weights
def visualize_attention(image, attention_weights):
    # Resize attention weights to match image size
    image_size = image.size(1)  # Assuming square images
    attention_map = F.interpolate(attention_weights.unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)[0]
    
    # Aggregate attention maps across all heads (optional)
    attention_map = attention_map.mean(dim=0)
    
    # Visualize attention heatmap on the image
    plt.imshow(image.permute(1, 2, 0))
    plt.imshow(attention_map, alpha=0.6, cmap='hot', interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Define your data directory and JSON file for evaluation
data_dir_eval = "path/to/evaluation_images"
json_file_eval = "path/to/evaluation_json_file.json"

# Create data loader for evaluation
eval_dataset = VQADataset(data_dir_eval, json_file_eval, tokenizer, transform)
eval_data_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

# Evaluate the VQA model
evaluate_vqa_model(vqa_model, eval_data_loader, device)
