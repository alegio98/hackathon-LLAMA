from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Carica il modello fine-tuned
model_path = "./llama_fine_tuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Creazione di un pipeline per la generazione di testo
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Prompt dettagliato
prompt = (
    "voglio andare da clodio a ostia "
    "Quali sono i bus disponibili e cosa consigli di fare?"
)

# Genera la risposta
result = generator(prompt, max_length=150, num_return_sequences=1)
print("Risposta generata:")
print(result[0]["generated_text"])
