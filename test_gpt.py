from gpt_handler import get_gpt_response

while True:

    text = input("You: ")

    if text == "exit":
        break

    reply = get_gpt_response(text)

    print("Bot:", reply)