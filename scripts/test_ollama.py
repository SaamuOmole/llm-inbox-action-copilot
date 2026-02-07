from inboxcopilot.llm.providers import OllamaProvider

p = OllamaProvider(model="llama3.1:8b")
print(p.generate("Return EXACTLY the word: OK"))