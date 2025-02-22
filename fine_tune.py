# fine_tune.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def fine_tune_model(text):
    # Load a pre-trained model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Prepare data for fine-tuning
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=150)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save the fine-tuned model
    model.save_pretrained("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")