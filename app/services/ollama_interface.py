import ollama

response = ollama.chat(model='mistral', messages=[{
    "role": "user",
    "content": "What's the largest star in the galaxy that the largest star in the Orion belt is in?"
}],
                       stream=True,)

for chunk in response:
    print(chunk['message']['content'], end='', flush=True)