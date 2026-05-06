# Mental Health Chatbot — Emotion-Aware Conversational AI

> A two-stage AI pipeline that first *understands how you feel*, then responds accordingly — built using a fine-tuned transformer for emotion detection and a large language model for empathetic response generation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How It Actually Works — The Core Pipeline](#how-it-actually-works--the-core-pipeline)
3. [Repository Structure](#repository-structure)
4. [File-by-File Deep Dive](#file-by-file-deep-dive)
   - [dataset.csv](#datasetcsv)
   - [train.py](#trainpy)
   - [predict.py](#predictpy)
   - [gpt_handler.py](#gpt_handlerpy)
   - [main.py](#mainpy)
   - [ui_app.py](#ui_apppy)
   - [ui_utils.py](#ui_utilspy)
   - [test_gpt.py](#test_gptpy)
   - [emotion_model/](#emotion_model)
5. [Setup & Installation](#setup--installation)
6. [Running the Project](#running-the-project)
7. [Testing & Execution Screenshots](#testing--execution-screenshots)
8. [Design Decisions & Known Limitations](#design-decisions--known-limitations)
9. [Team Contributions](#team-contributions)
10. [Dependencies](#dependencies)

---

## Project Overview

Mental health conversations are different from ordinary chatbot interactions. The user isn't just asking a question — they're often expressing something emotionally charged, and a generic reply ("I'm here to help!") can feel hollow or even dismissive. This project tries to fix that by making the chatbot *aware of emotional context* before it responds.

The approach is a two-stage pipeline:

1. **Emotion Detection** — A fine-tuned DistilRoBERTa model reads the user's message and classifies it into one of six emotion categories: `anger`, `anxiety`, `depression`, `joy`, `worry`, or `suicide`.

2. **Emotion-Conditioned Response Generation** — Based on the detected emotion, a specific behavioral instruction is injected into the prompt sent to a large language model (Llama 3.3-70B via Groq). The LLM then generates a response that's tonally appropriate for that emotional state.

The result: a bot that doesn't just *answer* — it responds with the right *register*. It doesn't give a cheerful reply when someone is describing depression. It doesn't offer generic coping tips when someone's venting frustration. The response is shaped by what the user is actually feeling.

The project has two interfaces: a command-line version (`main.py`) and a Streamlit web UI (`ui_app.py`).

---

## How It Actually Works — The Core Pipeline

Understanding the pipeline end-to-end before diving into individual files makes everything else much clearer.

```
User Input (text)
      │
      ▼
┌─────────────────────────────────────┐
│  predict.py — Emotion Detection     │
│  Fine-tuned DistilRoBERTa           │
│  → anger / anxiety / depression /   │
│    joy / worry / suicide            │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  main.py — Instruction Routing      │
│  Looks up emotion_instructions dict │
│  Constructs a context-rich prompt   │
│  with: detected emotion, behavior   │
│  rules, conversation history        │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  gpt_handler.py — LLM Response      │
│  Calls Groq API → Llama 3.3-70B     │
│  Returns empathetic, contextual     │
│  reply to the user                  │
└─────────────────────────────────────┘
      │
      ▼
Bot Reply Displayed to User
```

The model was trained from scratch on a labeled CSV dataset using HuggingFace's `Trainer` API. The base checkpoint is `j-hartmann/emotion-english-distilroberta-base`, a DistilRoBERTa model pre-trained on emotion data — we fine-tuned the classification head on our own dataset with the six target emotion labels.

---

## Repository Structure

```
Mental-health-chatbot/
│
├── emotion_model/              # Saved fine-tuned model + tokenizer + label encoder
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer files...
│   └── label_encoder.pkl
│
├── dataset.csv                 # Labeled training data (Emotion, Sentence)
├── train.py                    # Model fine-tuning script
├── predict.py                  # Inference — predicts emotion from input text
├── gpt_handler.py              # Groq/LLaMA API wrapper for response generation
├── main.py                     # CLI chatbot — the full conversation loop
├── ui_app.py                   # Streamlit-based web UI
├── ui_utils.py                 # Emotion → color mapping (UI display utility)
├── test_gpt.py                 # Quick test script for the GPT handler
├── app.py                      # Placeholder / entry point (currently empty)
└── .gitignore
```

---

## File-by-File Deep Dive

### `dataset.csv`

This is where everything starts. The CSV has two columns:

```
Emotion, Sentence
```

Each row is a labeled training example — a sentence paired with the emotion it expresses. The dataset covers six emotional categories: `anger`, `anxiety`, `depression`, `joy`, `worry`, and `suicide`.

**What the team understood about this file:**
The label distribution matters a lot here. If one emotion class (say, `joy`) has ten times more samples than `depression`, the model will be biased toward predicting `joy` more often. Ideally, you'd want roughly balanced classes or apply class weights during training. The team preprocessed the labels with `.str.lower().str.strip()` to handle any inconsistencies in casing or whitespace — a small but important data hygiene step that prevents the label encoder from treating `"Anger"` and `"anger"` as two different classes.

Empty rows are dropped using `df.dropna()`, and all text is lowercased to ensure consistency between training and inference. Since the tokenizer will handle most normalization, the main thing this step buys you is uniformity in the label space.

---

### `train.py`

This is the most technically dense file in the project and the one most worth understanding in full.

**What it does:**

The script fine-tunes `j-hartmann/emotion-english-distilroberta-base` — a DistilRoBERTa model that was pre-trained on a multi-label emotion corpus — on the team's custom dataset. DistilRoBERTa is a smaller, faster distilled version of RoBERTa, which itself is a robustly optimized BERT. For a classification task like emotion detection, it hits a good balance between accuracy and inference speed.

**Step-by-step breakdown:**

**1. Data Loading & Preprocessing**

```python
df = pd.read_csv("dataset.csv")
df.columns = ["Emotion","Sentence"]
df["Emotion"] = df["Emotion"].str.lower().str.strip()
```

Column names are forcibly renamed after loading — this defends against any header variations in the CSV. The label cleaning is belt-and-suspenders defensive coding.

**2. Label Encoding**

```python
le = LabelEncoder()
df['label'] = le.fit_transform(df['Emotion'])
pickle.dump(le, open("label_encoder.pkl","wb"))
```

`LabelEncoder` maps the string emotion labels to integer indices (e.g., `anger → 0`, `anxiety → 1`, etc.). The encoder is pickled immediately after fitting because you need the *exact same mapping* at inference time. If you re-fit the encoder on a different subset of data during prediction, the label-to-index mapping could shift and your predictions would be wrong. Saving it to disk and loading it in `predict.py` is the right call.

**3. Train/Test Split**

```python
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Sentence"], df["label"], test_size=0.2, random_state=42
)
```

20% held out for evaluation. `random_state=42` ensures reproducibility — you can re-run training and get the same split every time, which matters when comparing experiments.

**4. Tokenization**

```python
tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
```

`max_length=128` is a reasonable cap for sentence-level emotion data — most emotional statements don't require 512 tokens. Truncation handles the rare long input, and `padding=True` ensures all sequences in a batch are the same length (required for batched tensor operations).

**5. Custom Dataset Class**

```python
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.reset_index(drop=True)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
```

This wraps the tokenized data into a PyTorch `Dataset` object compatible with HuggingFace's `Trainer`. The `.reset_index(drop=True)` call is critical — after `train_test_split`, pandas retains original row indices. If you don't reset, accessing `self.labels[0]` might actually return the item at index 0 of the original dataframe, not the split. This is a subtle bug that trips a lot of people up.

**6. Model Loading**

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "j-hartmann/emotion-english-distilroberta-base",
    num_labels=len(le.classes_),
    ignore_mismatched_sizes=True
)
```

`ignore_mismatched_sizes=True` is the key flag here. The pre-trained model's classification head was built for a different number of output classes. By setting this flag, HuggingFace replaces the original classifier head with a new randomly-initialized one sized to `len(le.classes_)` (your 6 emotion categories), while keeping all the pre-trained transformer weights intact. This is standard practice for fine-tuning.

**7. Training Configuration**

```python
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch"
)
```

`2e-5` is a standard fine-tuning learning rate for transformer models — high enough to adapt the classification head, low enough not to catastrophically forget the pre-trained representations. 3 epochs is a reasonable starting point; too many and you'll overfit on a small dataset.

**8. Training & Saving**

```python
trainer.train()
trainer.evaluate()
model.save_pretrained("emotion_model")
tokenizer.save_pretrained("emotion_model")
```

Both the model and tokenizer are saved to the `emotion_model/` directory. Always save the tokenizer alongside the model — they're a matched pair and the tokenizer's vocabulary and special token configuration needs to match what the model was trained on.

> ⚠️ **Note:** `train.py` was originally developed in Google Colab (you can see the Colab header in the file's docstring). If you're running it locally, make sure the `dataset.csv` path and `emotion_model/` output directory are correct relative to where you run the script.

---

### `predict.py`

This is the inference module. It loads the trained model from disk and exposes a single `predict_emotion(text)` function that everything else calls.

**Core logic:**

```python
MODEL_PATH = "emotion_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
le = pickle.load(open("emotion_model/label_encoder.pkl", "rb"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
```

Models are loaded at module-import time (not inside the function), which means they're loaded once and reused across all prediction calls. This is important — loading a transformer model from disk on every prediction would add several seconds of latency per message.

`model.eval()` switches off dropout layers and other training-only behaviors, ensuring deterministic, production-correct inference.

**The prediction function:**

```python
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits).item()
    emotion = le.inverse_transform([prediction])[0]
    return emotion
```

`torch.no_grad()` disables gradient computation during inference, which speeds up the forward pass and reduces memory usage. `outputs.logits` is a raw score vector — one value per emotion class. `torch.argmax` picks the index of the highest score, and `le.inverse_transform` maps it back to the human-readable label (`"anger"`, `"joy"`, etc.).

One thing worth noting: this function returns the single most likely emotion class — there's no confidence score or threshold. If you wanted to add a fallback for low-confidence predictions ("I'm not sure what you're feeling"), you'd need to run `softmax` on the logits and check if the max probability clears some threshold.

The script also has a standalone test mode at the bottom:
```python
if __name__ == "__main__":
    while True:
        text = input("Enter sentence: ")
        emotion = predict_emotion(text)
        print("Emotion:", emotion)
```

Useful for quickly testing the model in isolation without starting the full chatbot.

---

### `gpt_handler.py`

Small file, but it's the bridge to the LLM backbone of the whole system.

```python
from groq import Groq

API_KEY = "YOUR_GROQ_API_KEY"
client = Groq(api_key=API_KEY)

def get_gpt_response(user_input):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful and empathetic mental health assistant. Give supportive and safe responses."
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].message.content
```

**What the team understood here:**

The model is `llama-3.3-70b-versatile` — Meta's Llama 3.3 70B parameter model, accessed via [Groq](https://groq.com), which provides extremely fast LLM inference through custom LPU hardware. Groq's free tier is generous enough to prototype with.

`temperature=0.7` sits in a useful middle ground: lower than the default (which would feel more predictable), but not so low that the bot becomes robotic. For mental health conversations, you want the model to vary its language naturally while staying coherent. `max_tokens=500` caps the response length — which keeps replies focused and prevents the model from writing essays.

> ⚠️ **Important:** The API key is hardcoded in this file as a string. In any real deployment, replace this with `os.environ.get("GROQ_API_KEY")`. Committing API keys to version control is a security risk.

---

### `main.py`

This is the CLI chatbot — the full working implementation of the two-stage pipeline.

**How the conversation loop works:**

```python
while True:
    user_text = input("You: ")
    if user_text.lower() == "exit":
        break

    # Stage 1: Emotion Detection
    emotion = predict_emotion(user_text)

    # Stage 2: Get behavior instruction for this emotion
    instruction = emotion_instructions.get(emotion, "Respond kindly and supportively.")

    # Stage 3: Safety check
    if emotion == "suicide":
        print("Bot: If you are in immediate danger, please consider reaching out...")

    # Stage 4: Build the full prompt
    prompt = f"""
    You are a safe, supportive and empathetic mental health assistant.
    Detected emotion: {emotion}
    Behavior instruction: {instruction}
    Conversation history: {chat_history}
    User message: {user_text}
    Respond appropriately with empathy and emotional intelligence.
    """

    # Stage 5: Get and display response
    reply = get_gpt_response(prompt)
    chat_history.append(f"User: {user_text}")
    chat_history.append(f"Bot: {reply}")

    # Limit memory window to last 10 messages
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    print("Bot:", reply)
```

**The `emotion_instructions` dictionary is the heart of the behavioral routing system:**

```python
emotion_instructions = {
    "anger": """Respond calmly and help the user process their anger.
    Encourage pausing before reacting.
    Suggest healthy ways to release anger like breathing or taking space.""",

    "anxiety": """Respond in a calm and reassuring tone.
    Suggest grounding or breathing techniques.
    Help the user feel safe and supported.""",

    "depression": """Respond with empathy and emotional validation.
    Encourage small achievable steps.
    Avoid toxic positivity.
    Offer gentle encouragement.""",

    "joy": """Respond positively and warmly.
    Encourage the user's positive emotional state.
    Reinforce healthy positive thinking.""",

    "suicide": """Respond very carefully and supportively.
    Express concern and empathy.
    Encourage seeking professional help.
    Suggest contacting trusted people.
    Never provide harmful suggestions.""",

    "worry": """Respond reassuringly.
    Help the user manage overthinking.
    Suggest practical coping techniques."""
}
```

This is a deliberate prompt-engineering technique. Rather than hoping the LLM will respond appropriately based on a generic system prompt, the behavior is explicitly conditioned per detected emotion. The LLM is, in effect, given a different "personality configuration" for each emotional state. The `suicide` category has the strongest guardrails — the system immediately prints a safety notice in the UI before even calling the LLM.

**Conversation memory:**

The `chat_history` list accumulates turns and is injected into every subsequent prompt, giving the LLM context about what was said earlier. This is a basic form of in-context memory — the model isn't stateful on its own; the history is passed along manually. The window is capped at the last 10 messages to prevent the prompt from growing indefinitely (which would eventually exceed the model's context window and add latency).

---

### `ui_app.py`

The Streamlit version of the chatbot — same logic as `main.py`, but wrapped in a web interface.

```python
import streamlit as st

@st.cache_resource
def load_models():
    from predict import predict_emotion
    from gpt_handler import get_gpt_response
    return predict_emotion, get_gpt_response

predict_emotion, get_gpt_response = load_models()
```

The `@st.cache_resource` decorator is doing important work here. Streamlit reruns the entire script on every user interaction (it's reactive by design). Without caching, the transformer model would be reloaded from disk on every single keypress. `@st.cache_resource` ensures the model is loaded once and held in memory for the lifetime of the session.

The imports are placed *inside* the `load_models()` function intentionally — this defers the heavy model loading until after the Streamlit page has rendered, so the user sees the UI before the model is loaded.

**Streamlit's `session_state` for conversation memory:**

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, message in st.session_state.messages:
    st.chat_message(role).write(message)
```

`st.session_state` persists across reruns within the same user session, which makes it the right place to store conversation history. Without it, every message would reset the conversation.

The full response cycle on new input:
```python
user_text = st.chat_input("Type your message")
if user_text:
    st.chat_message("user").write(user_text)
    emotion = predict_emotion(user_text)
    prompt = f"..."  # Construct emotion-aware prompt
    reply = get_gpt_response(prompt)
    st.session_state.messages.append(("user", user_text))
    st.session_state.messages.append(("assistant", reply))
    st.chat_message("assistant").write(reply)
```

Notably, the Streamlit UI has a slightly simplified prompt compared to `main.py` (no per-emotion instruction routing). Bringing the full `emotion_instructions` dictionary into `ui_app.py` would be a straightforward improvement that makes both interfaces behaviorally consistent.

---

### `ui_utils.py`

A small utility dictionary mapping emotion labels to display colors:

```python
emotion_colors = {
    "anger": "red",
    "anxiety": "orange",
    "depression": "purple",
    "joy": "green",
    "suicide": "red",
    "worry": "yellow"
}
```

This is intended to visually indicate the detected emotion in the UI (e.g., displaying a colored badge or label next to the message). The `st.info(f"Detected emotion: {emotion}")` line in `ui_app.py` is currently commented out — enabling it alongside `emotion_colors` would give users real-time visibility into what the system has detected.

---

### `test_gpt.py`

A focused test script for validating the GPT handler in isolation. Useful for checking API connectivity and response formatting without running the full chatbot loop. If you're setting up the project and need to verify your Groq API key works before training the model, run this first.


---

### `emotion_model/`

The saved output of `train.py`. Contains:
- The fine-tuned model weights (`model.safetensors` or `pytorch_model.bin`)
- Tokenizer files (vocabulary, config, merges)
- `label_encoder.pkl` — the pickled `LabelEncoder` that maps class indices back to emotion strings

This directory is loaded by `predict.py` at startup. If you clone the repository and plan to skip training, you'll need this directory pre-populated — either by running `train.py` yourself or obtaining the trained weights separately.

---

## Setup & Installation

**Prerequisites:**
- Python 3.9+
- pip
- A [Groq API key](https://console.groq.com) (free tier available)
- GPU recommended for training (Colab T4 works fine), CPU is sufficient for inference

**1. Clone the repository**

```bash
git clone https://github.com/Rajathanu/Mental-health-chatbot.git
cd Mental-health-chatbot
```

**2. Create a virtual environment (strongly recommended)**

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

**3. Install dependencies**

```bash
pip install torch transformers scikit-learn pandas groq streamlit
```

If you're on a CUDA-enabled GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**4. Set your Groq API key**

Open `gpt_handler.py` and replace:
```python
API_KEY = "YOUR_GROQ_API_KEY"
```
with your actual key. Or better, use an environment variable:
```python
import os
API_KEY = os.environ.get("GROQ_API_KEY")
```
Then in your terminal: `export GROQ_API_KEY=your_actual_key`

---

## Running the Project

### Step 1: Train the Model

> Skip this step if you already have the `emotion_model/` directory with saved weights.

```bash
python train.py
```

This will load and preprocess `dataset.csv`, tokenize the sentences, fine-tune the model for 3 epochs, and save everything to `emotion_model/`. Training on CPU will be slow (20–60 minutes depending on dataset size). Google Colab with a T4 GPU is the recommended path — the script was originally developed there.

**Expected output:**
```
{'loss': 0.xxxx, 'epoch': 1.0}
{'eval_loss': 0.xxxx, 'eval_accuracy': 0.xx, 'epoch': 1.0}
...
MODEL TRAINED SUCCESSFULLY
```

### Step 2: Run the CLI Chatbot

```bash
python main.py
```

**Expected output:**
```
Mental Health Support Bot Started
Type 'exit' to stop

You: I haven't been able to get out of bed all week
Bot: I hear you, and I want you to know that what you're feeling is valid...
```

Type `exit` to end the session. If you uncomment the debug print line in `main.py`, you'll also see the detected emotion printed above each bot response.

### Step 3: Run the Streamlit UI

```bash
streamlit run ui_app.py
```

Streamlit will open a browser window at `http://localhost:8501`. The first load takes a few seconds as the model is loaded into memory — subsequent messages are much faster.

### Quick Emotion Test (predict.py standalone)

```bash
python predict.py
```

```
Enter sentence: I feel like nobody cares about me
Emotion: depression

Enter sentence: Everything is going really well today!
Emotion: joy
```

---


## Design Decisions & Known Limitations

**Why DistilRoBERTa and not a simpler classifier?**
Emotion in text is highly context-dependent. "I'm done" can mean joy (finishing a project) or despair. Classical classifiers (Naive Bayes, SVM on TF-IDF) tend to miss this nuance. Transformer models capture long-range contextual meaning, which is what you want for emotion detection.

**Why Groq/LLaMA instead of OpenAI?**
Groq's inference speed is remarkably fast — responses often come back in under a second. For a conversational application, latency matters a lot. LLaMA 3.3-70B at Groq speeds performs comparably to GPT-4 class models on empathetic dialogue, and the free tier is sufficient for a project like this.

**Conversation memory is shallow.** The `chat_history` list passed to the LLM is a simple string accumulation. For longer sessions, older turns could push newer ones out of the context window. A proper solution would use a vector store or a summarization step to maintain meaningful long-term memory.

**No confidence thresholding on emotion predictions.** If the model is unsure (e.g., a neutral sentence like "okay"), it still picks the highest-scoring class. Adding a minimum confidence threshold and a fallback "neutral" category would make the system more robust in edge cases.

**The API key is hardcoded.** Fine for a prototype/academic project, but must be changed before any public deployment.

**`ui_app.py` doesn't use the full `emotion_instructions` routing.** The Streamlit UI sends a simpler prompt than `main.py`. Bringing the two into parity would be a clean improvement.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch — model training and inference |
| `transformers` | HuggingFace Transformers — DistilRoBERTa model, tokenizer, Trainer API |
| `scikit-learn` | LabelEncoder, train_test_split, accuracy_score |
| `pandas` | Dataset loading and preprocessing |
| `numpy` | Numerical operations |
| `groq` | Groq Python SDK — LLaMA 3.3-70B API access |
| `streamlit` | Web UI framework |
| `pickle` | Serializing/deserializing the label encoder |

Install all at once:
```bash
pip install torch transformers scikit-learn pandas numpy groq streamlit
```

---

*This project was built as an academic exploration of emotion-aware conversational AI. It is not a substitute for professional mental health support. If you or someone you know is in crisis, please contact a mental health professional or a crisis helpline.*
