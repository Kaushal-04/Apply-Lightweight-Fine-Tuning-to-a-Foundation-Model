from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import load_dataset

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)






# Load the dataset
dataset = load_dataset("imdb")

# Use a subset (e.g., first 1000 samples from the test set)
subset_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized_dataset = subset_dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")





# Create DataLoader
batch_size = 8
dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)

# Evaluate the model
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for batch in dataloader:
        inputs = {key: value.to(device) for key, value in batch.items() if key != "label"}
        labels = batch["label"].to(device)
        
        outputs = model(**inputs)
        predictions = torch.argmax(F.softmax(outputs.logits, dim=-1), dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")
