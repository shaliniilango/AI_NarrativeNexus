import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

st.title("🧠 Text Summarization App")

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

text = st.text_area("Enter text to summarize:")

if st.button("Summarize"):
    if text:
        input_text = "summarize: " + text
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summary:")
        st.write(summary)
