import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import pytz
import time

# Configura Streamlit
st.set_page_config(
    page_title="AI Voyager",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Carica il modello e il tokenizer
model_path = "./llama_fine_tuned"  # Percorso del modello fine-tuned
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Stile della pagina
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap'); /* Importa il font Poppins */
    .main {
        background-color: #f0f8ff;
    }
    .title {
        font-family: 'Poppins', sans-serif;
        color: #1e3a8a;
        text-align: center;
        font-size: 3em;
        font-weight: 600;
        margin-top: 20px;
    }
    .subtitle {
        font-family: 'Poppins', sans-serif;
        color: #374151;
        text-align: center;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .prompt-container {
        margin-top: 30px;
    }
    .generate-button {
        background-color: #2563eb;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Titolo
st.markdown("<h1 class='title'>AI Voyager 🌍</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p class='subtitle'>
        🇮🇹 Explore AI-powered text generation for urban mobility insights in the capital of Italy, Rome! 🏛️
    </p>
    """,
    unsafe_allow_html=True
)

# Input dell'utente
with st.container():
    st.markdown("<div class='prompt-container'>", unsafe_allow_html=True)
prompt = st.text_area(
    "🌐 Enter your prompt below:",
    placeholder=(
        "Example questions:\n"
        "- Traffic to Ostia Lido now\n"
        "- Can I leave at 8:00 AM and arrive at Piazza Navona on time without traffic?\n"
        "- I need to be at my office in Clodio by 10:00 AM. Any suggestions?\n"
        "- What's the fastest way to travel from Rome Termini to the Colosseum at 9:00 AM?"
    ),
)
st.markdown("</div>", unsafe_allow_html=True)

# Parametri per la generazione
st.sidebar.header("Generation Settings")
max_length = st.sidebar.slider("📏 Max Length", min_value=10, max_value=150, value=150, step=10)
top_k = st.sidebar.slider("🎲 Top-K Sampling", min_value=0, max_value=100, value=50, step=10)


# Genera testo al clic su un bottone
if st.button("🚀 Ask the Chatbot", key="generate"):
    if not prompt.strip():
        st.warning("Please enter a valid prompt to generate text.")
    else:
        with st.spinner("AI is crafting your response..."):
            # Tokenizza l'input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Genera testo
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                top_k=top_k,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            # Decodifica e mostra il risultato
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success("Generated text successfully!")
            st.write("### ✨ Here's your generated text:")
            st.write(generated_text)

# Separatore con icona e linea
st.markdown(
    """
    <hr style="border: none; border-top: 3px solid #2563eb; margin: 20px 0;">
    <div style="text-align: center; font-size: 24px; font-family: 'Poppins', sans-serif; color: #2563eb;">
        🚦 Insights for Urban Mobility in Rome <span style="color: #e63946;">(LIVE 📡)</span>
    </div>
    <div style="text-align: center; font-size: 16px; font-family: 'Poppins', sans-serif; color: #374151; margin-top: 10px;">
        <span id="timer"></span>
    </div>
    <hr style="border: none; border-top: 3px solid #2563eb; margin: 20px 0;">
    """,
    unsafe_allow_html=True,
)


# Vie più trafficate
st.markdown("### 🚦 Most Congested Roads to Avoid")
congested_roads = [
    "Via Flaminia (North Rome) - Peak hours: 8:00 AM - 10:00 AM",
    "Via Tiburtina (East Rome) - Peak hours: 5:00 PM - 7:00 PM",
    "Lungotevere (Riverside) - Constant delays during the day",
]
for road in congested_roads:
    st.write(f"- {road}")

# Bus con ritardo, live on model
st.markdown("### 🚌 Buses with the Most Delays during the day")
delayed_buses = [
    "Bus 64: 15 min delay on average",
    "Bus 492: Frequently delayed during peak hours",
    "Bus 87: Significant delays due to construction works",
]
for bus in delayed_buses:
    st.write(f"- {bus}")

# Qualità dell'aria
st.markdown("### 🌬️ Best and Worst Air Quality Areas")
best_air_quality = "Villa Borghese (AQI: 35 - Good)"
worst_air_quality = "Piazza Venezia (AQI: 120 - Unhealthy for Sensitive Groups)"
st.write(f"- **Best Air Quality:** {best_air_quality}")
st.write(f"- **Worst Air Quality:** {worst_air_quality}")


# Footer
st.markdown(
    """
    <hr>
    <footer style="text-align: center; font-size: 14px; margin-top: 20px; color: #6b7280;">
        Made with ❤️ by <strong>Team Hack4Hope</strong>
    </footer>
    """,
    unsafe_allow_html=True,
)

