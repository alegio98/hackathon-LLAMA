# **Hack4Hope Project: "SmartMove" - Mobilità Intelligente per una Vita Sostenibile**

## **Overview**
"SmartMove" è un assistente digitale basato sull’intelligenza artificiale, progettato per ottimizzare la mobilità urbana a Roma. Utilizzando dati sul traffico, la qualità dell’aria e gli orari di punta, SmartMove suggerisce i momenti migliori per spostarsi, offrendo alternative ecologiche e percorsi ottimali.

Il progetto combina modelli avanzati di AI, come GPT-2, con dati provenienti da fonti ufficiali come il Comune di Roma e l’Agenzia per l’Italia Digitale (AGID), per creare un'applicazione pratica e user-friendly.

---

## **Notebooks**
1. **Data Preprocessing & Transformation**:
   - Ampio web scraping per il recupero da dataset incompleti + uso di dataset ufficiali e revisionati di lablab.ai
   - Normalizzazione e preparazione del dataset proveniente da fonti ufficiali.
   - Generazione di frasi contestuali in linguaggio naturale basate sui dati sul traffico e sull'inquinamento.
   
3. **Fine-Tuning del Modello**:
   - Addestramento di un modello GPT-2 ottimizzato su frasi generate dal dataset.
   - Parametri personalizzati per l’addestramento e strategie di ottimizzazione.

4. **Model Deployment Pipeline**:
   - Integrazione del modello fine-tuned con un’interfaccia Streamlit.
   - Test e validazione del sistema di generazione testuale.

---

## **Requirements**
### **Prerequisiti**
- **Python Version**: 3.10+
- **Librerie Necessarie**:
  - `Transformers`
  - `PyTorch`
  - `Datasets`
  - `Streamlit`
  - `scikit-learn`


---
## **Datasets and Models**

### **Fonti dei Dati**
- **LabLabAI official datasets**: ( https://lablab.ai/tech/llama-meta-rome-datasets )
- **Comune di Roma**: Dati sul traffico e la mobilità urbana.
- **Agenzia per l’Italia Digitale (AGID)**: Dataset aperti sull'inquinamento e sui trasporti pubblici.
- **OpenAQ**: Livelli di qualità dell’aria (PM10, NO2, O3, AQI) in tempo reale.

### **Modello Utilizzato**
- **GPT-2 (pre-trained)**:
  - Fine-tuning specifico per l’analisi del traffico e la qualità dell’aria.

---

## **Features**

### **1. Ottimizzazione della Mobilità Urbana**
- Suggerimenti personalizzati basati su:
  - Traffico, velocità media e rallentamenti.
  - Alternative ecologiche come autobus e percorsi pedonali.

### **2. Consapevolezza Ambientale**
- Indicazioni sulla qualità dell’aria in tempo reale.
- Suggerimenti per evitare zone ad alto inquinamento.

### **3. Tecnologie Avanzate**
- Fine-tuning di GPT-2 per generare frasi contestuali in inglese e italiano.
- Interfaccia Streamlit per interazioni user-friendly.

---

## **Contributors**

### **Team Hack4Hope**
- Alessandro Giovannini
- Floriano Caprio
- Davide M. Bozzone

---

## **Acknowledgements**
- **Comune di Roma**: Per i dati sul traffico urbano.
- **AGID (Agenzia per l’Italia Digitale)**: Per i dataset aperti su trasporti e mobilità.
- **OpenAQ**: Per i dati sulla qualità dell’aria.
- **Hugging Face**: Per i modelli GPT-2 pre-addestrati e le librerie di machine learning.

---

## **Future Work**
- Espansione del progetto per includere più città italiane.
- Integrazione con sistemi di trasporto pubblico in tempo reale.
- Supporto a più lingue per espandere l’usabilità internazionale.


