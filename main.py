from predict import predict_emotion
from gpt_handler import get_gpt_response

print("\nMental Health Support Bot Started")
print("Type 'exit' to stop\n")

# ========================
# Conversation Memory
# ========================
chat_history = []

# ========================
# Emotion Instructions
# ========================
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

# ========================
# Main Chat Loop
# ========================
while True:

    user_text = input("You: ")

    # Exit condition
    if user_text.lower() == "exit":

        print("\nBot: Take care. Remember you are not alone.\n")
        break

    # ========================
    # Step 1: Emotion Detection
    # ========================
    emotion = predict_emotion(user_text)

    # print(f"[Detected emotion → {emotion}]")

    # ========================
    # Step 2: Get Instruction
    # ========================
    instruction = emotion_instructions.get(
        emotion,
        "Respond kindly and supportively."
    )

    # ========================
    # Step 3: Suicide Safety Notice
    # ========================
    if emotion == "suicide":

        print("Bot: If you are in immediate danger, please consider reaching out to a trusted person or mental health professional.")

    # ========================
    # Step 4: Prompt Creation
    # ========================
    prompt = f"""
You are a safe, supportive and empathetic mental health assistant.

Detected emotion: {emotion}

Behavior instruction:
{instruction}

Conversation history:
{chat_history}

User message:
{user_text}

Respond appropriately with empathy and emotional intelligence.
"""
    # ========================
    # Step 5: GPT Response
    # ========================
    reply = get_gpt_response(prompt)
    # ========================
    # Step 6: Save Memory
    # ========================
    chat_history.append(f"User: {user_text}")
    chat_history.append(f"Bot: {reply}")
    # Limit conversation memory
    if len(chat_history) > 10:

        chat_history = chat_history[-10:]
    # ========================
    # Step 7: Display Reply
    # ========================
    print("Bot:", reply)