import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import torch.nn.functional as F
from torch.utils.data import DataLoader

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
SAVE_DIR = "./peft_distilbert"

def load_base_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, load_in_4bit=True  # Automatically handles device placement
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def load_data():
    dataset = load_dataset("imdb")
    # Split a portion of the training data into validation data
    train_data = dataset["train"].shuffle(seed=42).select(range(2000))
    val_data = dataset["test"].shuffle(seed=42).select(range(1000))  # Use the test set as validation
    return train_data, val_data

def apply_peft(model):
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_lin"], bias="none", task_type="SEQ_CLS"
    )
    return get_peft_model(model, lora_config)

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)

def fine_tune_model(peft_model, tokenized_train_dataset, tokenized_eval_dataset):
    training_args = TrainingArguments(
        output_dir=SAVE_DIR, 
        per_device_train_batch_size=8, 
        num_train_epochs=1,
        logging_dir="./logs", 
        save_strategy="epoch", 
        learning_rate=2e-4, 
        fp16=True, 
        evaluation_strategy="epoch"  # Will evaluate the model after each epoch
    )
    trainer = Trainer(
        model=peft_model, 
        args=training_args, 
        train_dataset=tokenized_train_dataset, 
        eval_dataset=tokenized_eval_dataset  # Pass the eval dataset for evaluation
    )
    trainer.train()
    peft_model.save_pretrained(SAVE_DIR)

def load_fine_tuned_model():
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, load_in_4bit=True)
    model = PeftModel.from_pretrained(base_model, SAVE_DIR)
    return model

def evaluate_model(model, tokenized_eval_dataset):
    dataloader = DataLoader(tokenized_eval_dataset, batch_size=8)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: value for key, value in batch.items() if key != "label"}
            labels = batch["label"]

            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    model, tokenizer = load_base_model()
    train_data, val_data = load_data()

    # Tokenize the datasets
    tokenized_train_data = train_data.map(lambda x: tokenize_function(x, tokenizer), batched=True).remove_columns(["text"])
    tokenized_val_data = val_data.map(lambda x: tokenize_function(x, tokenizer), batched=True).remove_columns(["text"])

    tokenized_train_data.set_format("torch")
    tokenized_val_data.set_format("torch")

    # Apply PEFT to the model
    peft_model = apply_peft(model)

    # Fine-tune the model with the training and validation datasets
    fine_tune_model(peft_model, tokenized_train_data, tokenized_val_data)

    # Load the fine-tuned model and evaluate
    loaded_model = load_fine_tuned_model()
    evaluate_model(loaded_model, tokenized_val_data)
