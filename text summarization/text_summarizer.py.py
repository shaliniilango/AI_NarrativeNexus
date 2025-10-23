from transformers import T5ForConditionalGeneration, T5Tokenizer
# Load tokenizer and model
model_name = "t5-small"  # you can change to "t5-base" for better results
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
def summarize_text(text, max_length=150, min_length=40):
    # Preprocess input text for T5
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary (beam search for quality)
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode and return summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
text = """
Artificial Intelligence (AI) is transforming industries by automating complex tasks,
improving decision-making, and enabling the development of innovative applications.
From healthcare and finance to entertainment and education, AI’s potential seems limitless.
However, with its rapid growth, ethical considerations and responsible usage are becoming more important than ever.
"""

summary = summarize_text(text)
print("Original Text:\n", text)
print("\nGenerated Summary:\n", summary)
