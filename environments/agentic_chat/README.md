# ReAct Agent Environment

### Overview
- **Environment ID**: `agentic-chat`
- **Short description**: A multi-turn ReAct (Reasoning + Acting) agent environment where agents use tools iteratively and terminate by calling a `final_answer` tool
- **Tags**: multi-turn, tool-use, react, agent, reasoning

### What is This?

This environment implements the **ReAct pattern** for building agents that:

1. **Reason** about what information they need
2. **Act** by calling tools to gather information
3. **Iterate** between reasoning and acting
4. **Terminate** when they call the `final_answer` tool with their answer

Think of it as a loop where your agent can use tools like `search`, `calculator`, etc., and when it has enough information, it calls `final_answer(answer="...")` to end the loop.

### Quick Start

```bash
uv run vf-install agentic_chat
```

```bash
uv run vf-eval agentic_chat -m gpt-4.1-2025-04-14
```

### Key Features

- ✅ Automatic tool management (pass Python functions, get OpenAI tool format)
- ✅ Built-in `final_answer` tool for graceful termination
- ✅ Multiple termination conditions (final answer, max turns, no tool calls)
- ✅ State tracking (tool usage statistics, timing, etc.)
- ✅ Extensible and customizable

### Quick Start

See the complete example in [`example.py`](./example.py):

```bash
# Install dependencies
pip install verifiers openai datasets

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run the example
python example.py
```

### Basic Usage

```python
from agentic_chat import ReactAgentEnv

# Define tools
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

# Create environment
env = ReactAgentEnv(
    tools=[calculator],
    max_turns=10,
    system_prompt="You are a helpful assistant with access to tools."
)

# The environment automatically adds a final_answer tool
# Agents should call final_answer(answer="...") when done
```

### How the Loop Works

```
1. User asks a question
2. Agent reasons about what to do
3. Agent calls tools to get information
4. Tools return results
5. Agent sees results and reasons more
6. Repeat steps 3-5 until...
7. Agent calls final_answer(answer="...")
8. Loop terminates ✓
```

### Termination Conditions

The agent loop stops when **any** of these happen:

1. **Final Answer Called**: Agent calls `final_answer(answer="...")`
2. **Max Turns Reached**: `state["turn"] >= max_turns`
3. **No Tool Calls**: Agent responds without calling any tools
4. **Context Too Long**: Prompt exceeds model's context length

### Example Tools

```python
def search_database(query: str) -> str:
    """Search a database for information."""
    # Your implementation
    return results

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Your implementation
    return weather_info

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))
```

Tools must have:
- Type hints for parameters
- Docstring describing what they do
- Return a string (or JSON-serializable value)

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `tools` | `list[Callable]` | `[]` | List of Python functions to use as tools |
| `max_turns` | `int` | `10` | Maximum number of agent turns before termination |
| `include_final_answer_tool` | `bool` | `True` | Whether to automatically add the final_answer tool |
| `error_formatter` | `Callable` | `lambda e: f"Error: {e}"` | Function to format tool execution errors |
| `system_prompt` | `str` | `None` | System prompt to guide agent behavior |
| `dataset` | `Dataset` | `None` | Training dataset |
| `eval_dataset` | `Dataset` | `None` | Evaluation dataset |

### State Tracking

Each rollout tracks:

```python
state = {
    "turn": int,                    # Current turn number
    "final_answer": str | None,     # The final answer (if provided)
    "tool_usage": dict[str, int],   # Tool name -> call count
    "timing": {
        "generation_ms": float,     # Time spent generating
        "total_ms": float,          # Total time
    },
    # ... other fields
}
```

Access after evaluation:

```python
results = await env.evaluate(...)
for state in results.state:
    print(f"Turns: {state['turn']}")
    print(f"Final answer: {state['final_answer']}")
    print(f"Tools used: {state['tool_usage']}")
```

### Complete Documentation

For a comprehensive guide including:
- Detailed architecture explanation
- Advanced usage patterns
- Custom termination logic
- Tool execution hooks
- Custom reward functions
- Best practices and troubleshooting

See: **[REACT_AGENT_ENVIRONMENT_GUIDE.md](./REACT_AGENT_ENVIRONMENT_GUIDE.md)**

### Files in This Directory

- `agentic_chat.py` - The ReactAgentEnv implementation
- `example.py` - Complete working example
- `REACT_AGENT_ENVIRONMENT_GUIDE.md` - Comprehensive documentation
- `README.md` - This file (quick reference)

### Datasets

This environment is **dataset-agnostic**. You can use any dataset with:
- A `question` column (will be formatted as user message)
- An optional `answer` column (for evaluation)

Example:
```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "question": ["What is 2+2?", "Capital of France?"],
    "answer": ["4", "Paris"]
})

env.dataset = env.format_dataset(dataset)
```

### Metrics

The environment tracks these key metrics in the state:

| Metric | Meaning |
| ------ | ------- |
| `turn` | Number of turns taken |
| `tool_usage` | Dictionary of tool name → call count |
| `final_answer` | The agent's final answer (if provided) |
| `timing.generation_ms` | Time spent generating responses |

You can define custom reward functions to score agent behavior (see the comprehensive guide).

### Example Output

```
Example 1
-----------------------------------------------------------
Question: What is 25 multiplied by 17?
Expected Answer: 425
Agent's Final Answer: 425
Turns Used: 2
Tool Usage: {'calculator': 1, 'final_answer': 1}
Reward: 1.00
Generation Time: 1234.56ms
```

### Common Patterns

#### Simple Q&A Agent
```python
env = ReactAgentEnv(
    tools=[search_tool, calculator],
    max_turns=10,
    system_prompt="Use tools to answer questions accurately."
)
```

#### Research Agent
```python
env = ReactAgentEnv(
    tools=[wikipedia_search, arxiv_search, web_search],
    max_turns=20,
    system_prompt="Research thoroughly before answering."
)
```

#### Data Analysis Agent
```python
env = ReactAgentEnv(
    tools=[query_database, run_sql, plot_data],
    max_turns=15,
    system_prompt="Analyze data and provide insights."
)
```

### Tips

1. **Clear Instructions**: Tell the agent to call `final_answer` in the system prompt
2. **Good Tool Descriptions**: Write detailed docstrings for your tools
3. **Error Handling**: Wrap tool logic in try/except blocks
4. **Few-Shot Examples**: Show the agent how to use tools effectively
5. **Monitor Tool Usage**: Check `state['tool_usage']` to see what the agent is doing

### Troubleshooting

**Agent doesn't call final_answer?**
- Add explicit instruction: "Always call final_answer when you have the answer"
- Provide few-shot examples showing final_answer usage

**Tools not being called?**
- Ensure tools have docstrings and type hints
- Check that the model supports function calling (gpt-4, gpt-3.5-turbo, etc.)

**High latency?**
- Increase `max_concurrent` for parallel execution
- Use faster models for initial testing

See the comprehensive guide for more troubleshooting help.

---

**For complete documentation, examples, and advanced usage, see [REACT_AGENT_ENVIRONMENT_GUIDE.md](./REACT_AGENT_ENVIRONMENT_GUIDE.md)**
