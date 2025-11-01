import verifiers as vf # type: ignore
from datasets import Dataset # type: ignore


system_prompt="You are a helpful assistant with access to tools. Use the calculator for mathematical operations. Use search_database to find information about topics. Always call final_answer tool when you have the complete answer to the user's question."
dataset = Dataset.from_dict({
    "question": ["What is the capital of France?", "What is 25 multiplied by 17?", "Tell me about Python programming language"],
    "answer": ["Paris", "425", "Python is a high-level, interpreted programming language"],
})
vf_env = vf.load_environment("agentic-chat", tools=[], max_turns=10, system_prompt=system_prompt, dataset=dataset)


