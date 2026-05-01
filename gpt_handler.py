from groq import Groq
import os

# Always safer to use environment variable
API_KEY = "YOUR_GROQ_API_KEY"

# Create client
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