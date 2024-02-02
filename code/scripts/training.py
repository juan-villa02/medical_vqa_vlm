from sklearn.metrics import accuracy_score, f1_score
from transformers import VisualBertForQuestionAnswering, VisualBertTokenizer
import torch

# Load pre-trained model and tokenizer
model = VisualBertForQuestionAnswering.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')
tokenizer = VisualBertTokenizer.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')

# Assume validation_images, validation_questions, and validation_answers are your validation data
# They should be preprocessed in the same way as your training data

predictions = []

# Loop over validation data
for image, question in zip(validation_images, validation_questions):
    # Process image and question and feed to model
    inputs = tokenizer(question, return_tensors='pt')
    outputs = model(inputs, visual_embeds=image)

    # Get predicted answer and add to predictions list
    predicted_answer = tokenizer.decode(outputs.logits.argmax(dim=-1))
    predictions.append(predicted_answer)

# Calculate accuracy and F1 score
accuracy = accuracy_score(validation_answers, predictions)
f1 = f1_score(validation_answers, predictions, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')


from transformers import VisualBertForQuestionAnswering, VisualBertTokenizer
from torch.utils.data import DataLoader
import torch.optim as optim

# Load pre-trained model and tokenizer
model = VisualBertForQuestionAnswering.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')
tokenizer = VisualBertTokenizer.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')

# Assume images, questions, and answers are your training data
# They should be preprocessed in the same way as your validation data

# Create a DataLoader for your data
dataset = list(zip(images, questions, answers))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loop over epochs
for epoch in range(10):
    # Loop over batches
    for batch in dataloader:
        images, questions, answers = batch

        # Zero the gradients
        optimizer.zero_grad()

        # Process images and questions and feed to model
        inputs = tokenizer(questions, return_tensors='pt')
        outputs = model(inputs, visual_embeds=images)

        # Calculate loss
        loss = outputs.loss

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

    from transformers import VisualBertForQuestionAnswering, VisualBertTokenizer
    from PIL import Image
    import torch

    # Load fine-tuned model and tokenizer
    model = VisualBertForQuestionAnswering.from_pretrained('path_to_your_model')
    tokenizer = VisualBertTokenizer.from_pretrained('path_to_your_model')

    def predict(image_path, question):
        # Open and process image
        image = Image.open(image_path).convert('RGB')
        image = torch.tensor(image).unsqueeze(0)  # Assuming your preprocessing transforms the image into a tensor

        # Tokenize question
        inputs = tokenizer(question, return_tensors='pt')

        # Make prediction
        outputs = model(inputs, visual_embeds=image)

        # Get predicted answer
        predicted_answer = tokenizer.decode(outputs.logits.argmax(dim=-1))

        return predicted_answer

    # Use the function
    image_path = 'path_to_your_image'
    question = 'your_question'
    print(predict(image_path, question))


# Assume 'model' is your fine-tuned VisualBERT model
model_path = "path_to_save_model"
model.save_pretrained(model_path)

from transformers import VisualBertForQuestionAnswering

model_path = "path_to_saved_model"
model = VisualBertForQuestionAnswering.from_pretrained(model_path)



# Assume 'tokenizer' is your fine-tuned tokenizer
tokenizer_path = "path_to_save_tokenizer"
tokenizer.save_pretrained(tokenizer_path)


from transformers import VisualBertTokenizer

tokenizer_path = "path_to_saved_tokenizer"
tokenizer = VisualBertTokenizer.from_pretrained(tokenizer_path)


from transformers import VisualBertForQuestionAnswering, VisualBertTokenizer
from torch.utils.data import DataLoader
import torch.optim as optim

# Load pre-trained model and tokenizer
model = VisualBertForQuestionAnswering.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')
tokenizer = VisualBertTokenizer.from_pretrained('uclanlp/visualbert-nlvr2-coco-pre')

# Assume new_images, new_questions, and new_answers are your new training data
# They should be preprocessed in the same way as your validation data

# Create a DataLoader for your new data
dataset = list(zip(new_images, new_questions, new_answers))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Set up optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loop over epochs
for epoch in range(10):
    # Loop over batches
    for batch in dataloader:
        images, questions, answers = batch

        # Zero the gradients
        optimizer.zero_grad()

        # Process images and questions and feed to model
        inputs = tokenizer(questions, return_tensors='pt')
        outputs = model(inputs, visual_embeds=images)

        # Calculate loss
        loss = outputs.loss

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()


from sklearn.metrics import accuracy_score, f1_score

# Assume validation_images, validation_questions, and validation_answers are your validation data
# They should be preprocessed in the same way as your training data

predictions = []

# Loop over validation data
for image, question in zip(validation_images, validation_questions):
    # Process image and question and feed to model
    inputs = tokenizer(question, return_tensors='pt')
    outputs = model(inputs, visual_embeds=image)

    # Get predicted answer and add to predictions list
    predicted_answer = tokenizer.decode(outputs.logits.argmax(dim=-1))
    predictions.append(predicted_answer)

# Calculate accuracy and F1 score
accuracy = accuracy_score(validation_answers, predictions)
f1 = f1_score(validation_answers, predictions, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')




from transformers import VisualBertForQuestionAnswering, VisualBertTokenizer
from PIL import Image
import torch

# Load fine-tuned model and tokenizer
model = VisualBertForQuestionAnswering.from_pretrained('path_to_your_model')
tokenizer = VisualBertTokenizer.from_pretrained('path_to_your_model')

def vqa(image_path, question):
    # Open and process image
    image = Image.open(image_path).convert('RGB')
    image = torch.tensor(image).unsqueeze(0)  # Assuming your preprocessing transforms the image into a tensor

    # Tokenize question
    inputs = tokenizer(question, return_tensors='pt')

    # Make prediction
    outputs = model(inputs, visual_embeds=image)

    # Get predicted answer
    predicted_answer = tokenizer.decode(outputs.logits.argmax(dim=-1))

    return predicted_answer

# Use the VQA system
image_path = 'path_to_your_image'
question = 'your_question'
print(vqa(image_path, question))



