from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch

import pickle

# LOAD MODEL PATH
MODEL_PATH = "emotion_model"

# LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# LOAD MODEL
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# LOAD LABEL ENCODER
le = pickle.load(open("emotion_model/label_encoder.pkl","rb"))

# DEVICE (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.eval()

# PREDICTION FUNCTION
def predict_emotion(text):

    # TOKENIZE INPUT
    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        padding=True

    )

    inputs = inputs.to(device)

    # MODEL PREDICTION
    with torch.no_grad():

        outputs = model(**inputs)

    # GET PREDICTED CLASS
    prediction = torch.argmax(outputs.logits).item()

    # CONVERT TO EMOTION LABEL
    emotion = le.inverse_transform([prediction])[0]

    return emotion


# TEST BLOCK
if __name__ == "__main__":

    while True:

        text = input("Enter sentence: ")

        emotion = predict_emotion(text)

        print("Emotion:",emotion)