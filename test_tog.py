from together import Together

client = Together()

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello"}],
)

print(response.usage)