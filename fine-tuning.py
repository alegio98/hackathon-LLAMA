from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

# Step 1: Carica i dati
def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [{"text": line.strip()} for line in lines]

data_path = "llama_fine_tuning_examples.txt"  # Path to the fine-tuning examples file
data = load_data(data_path)

# Convert data to Hugging Face Dataset
dataset = Dataset.from_list(data)

# Step 2: Tokenizer e Modello
model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configura il tokenizer per aggiungere il token di padding
tokenizer.pad_token = tokenizer.eos_token  # Usa il token di fine sequenza (eos) come token di padding

# Tokenizza i dati
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./llama_fine_tuned",
    evaluation_strategy="steps",  # Valutazione durante il training
    eval_steps=500,  # Ogni 500 step
    learning_rate=3e-5,  # Riduci il learning rate per stabilità
    per_device_train_batch_size=16,  # Aumenta se possibile
    num_train_epochs=5,  # Più epoche
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,  # Precisione a 16-bit per accelerare
    logging_dir="./logs",
)

# classe Trainer personalizzatan richiesta
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs):
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels)
        return outputs.loss


train_data = tokenized_dataset
eval_data = tokenized_dataset

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,  # Usa lo stesso dataset
    tokenizer=tokenizer,
)



# Step 4: Addestramento
trainer.train()

# Step 5: Salva il modello fine-tuned
model.save_pretrained("./llama_fine_tuned")
tokenizer.save_pretrained("./llama_fine_tuned")
