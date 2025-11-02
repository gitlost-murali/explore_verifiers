"""
ReAct Agent Environment

This environment implements a ReAct (Reasoning + Acting) agent pattern where:
1. The agent can use tools to gather information
2. The agent reasons about the information
3. The loop continues until the agent calls a final_answer tool
"""

import json
from typing import Any, Callable

from openai import AsyncOpenAI
import verifiers as vf
from verifiers.types import Messages, State # type: ignore
from openai.types.chat import ChatCompletionMessageToolCall
from datasets import load_dataset
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class JudgeResponse(BaseModel):
    reasoning: str
    score: float

def final_answer(answer: str) -> str:
    """
    Call this tool when you have the final answer to the user's question.
    This will terminate the agent loop.

    Args:
        answer: The final answer to return to the user

    Returns:
        The answer string
    """
    return answer


class ReactAgentEnv(vf.ToolEnv):
    """
    ReAct Agent Environment

    A multi-turn environment where an agent can:
    - Use tools to gather information
    - Reason about the gathered information
    - Call final_answer to terminate the loop with a final answer

    The environment automatically terminates when:
    1. The agent calls the final_answer tool
    2. max_turns is reached
    3. The agent generates a message without tool calls (optional)
    """

    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        include_final_answer_tool: bool = True,
        error_formatter: Callable[[Exception], str] = lambda e: f"Error: {str(e)}",
        **kwargs,
    ):
        """
        Initialize the ReAct Agent Environment

        Args:
            tools: List of callable tools the agent can use
            max_turns: Maximum number of turns before termination
            include_final_answer_tool: Whether to automatically add final_answer tool
            error_formatter: Function to format tool execution errors
            **kwargs: Additional arguments passed to parent ToolEnv
        """
        tools = tools or []

        if include_final_answer_tool:
            tools = [final_answer] + tools

        super().__init__(
            tools=tools,
            max_turns=max_turns,
            error_formatter=error_formatter,
            **kwargs,
        )

        self.include_final_answer_tool = include_final_answer_tool

    async def is_completed(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> bool:
        """
        Check if the agent loop should terminate.

        Terminates when:
        1. max_turns is reached
        2. The agent generated a message without tool calls (inherited behavior)
        3. The agent called the final_answer tool
        """
        # Check parent completion conditions (max_turns, no tool calls)
        parent_completion_conditions = await super().is_completed(messages, state, **kwargs)
        final_answer = state.get("final_answer")

        return parent_completion_conditions or final_answer is not None

    async def setup_state(self, state: State, **kwargs) -> State:
        """
        Initialize state with ReAct-specific fields.
        """
        state = await super().setup_state(state, **kwargs)

        if "tool_usage" not in state:
            state["tool_usage"] = {}

        if "final_answer" not in state:
            state["final_answer"] = None

        return state

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ):
        """
        Override call_tool to track tool usage statistics.
        """
        tool_usage = kwargs.get("state", {}).get("tool_usage", {})
        tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

        if tool_name == "final_answer":
            kwargs["state"]["final_answer"] = tool_args.get("answer", None)

        return await super().call_tool(tool_name, tool_args, tool_call_id, **kwargs)

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> tuple[Messages, State]:
        """
        Execute tool calls and return tool responses.

        This is called after each assistant message that contains tool calls.
        """
        assert isinstance(messages, list)
        assert "tool_calls" in messages[-1]

        tool_messages = []
        for tool_call in messages[-1]["tool_calls"]:  # type: ignore
            match tool_call:
                case ChatCompletionMessageToolCall():
                    tool_name: str = tool_call.function.name
                    tool_args: dict = json.loads(tool_call.function.arguments)
                    tool_call_id: str = tool_call.id or ""
                case _:
                    assert "function" in tool_call
                    tool_name: str = tool_call["function"]["name"] # type: ignore
                    tool_args: dict = json.loads(tool_call["function"]["arguments"]) # type: ignore
                    tool_call_id: str = tool_call["id"] # type: ignore

            tool_message = await self.call_tool(
                tool_name, tool_args, tool_call_id, state=state
            )
            tool_messages.append(tool_message)

        return tool_messages, state

async def judge_response(prompt: str, completion: list[dict], answer: str, state: dict) -> float:
    response = completion[-1]['content']
    client = AsyncOpenAI()
    judge_prompt = f"""Evaluate the response based on the accuracy of the solution
    Score should be either 0.0 or 1.0.
    Response: {response}
    Answer: {answer}"""

    judge_result = await client.beta.chat.completions.parse(
        model="gpt-4.1-2025-04-14",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format=JudgeResponse
    )
    response = judge_result.choices[0].message.parsed
    return response.score if response else 0.0

def load_environment(**kwargs) -> vf.Environment:
    """
    Loads the ReAct Agent environment.

    Example usage:
        env = load_environment(
            tools=[search_tool, calculator_tool],
            max_turns=15,
            system_prompt="You are a helpful research assistant..."
        )

    Args:
        **kwargs: Arguments passed to ReactAgentEnv

    Returns:
        ReactAgentEnv instance
    """
    train_dataset = load_dataset(path="openai/gsm8k", name="main")["train"]
    eval_dataset = load_dataset(path="openai/gsm8k", name="main")["test"]
    system_prompt = "You are a helpful research assistant. You are given a question and you need to answer it using the tools provided to you. You need to call the final_answer tool when you have the final answer to the user's question."

    tools =  [final_answer] + kwargs.get("tools", [])
    tool_rubric = vf.ToolRubric(tools=tools)
    tool_rubric.reward_weights[1] = 0.2 # first tool is final_answer, so we give it a weight of 0.2
    rubric_group = vf.RubricGroup(rubrics=[tool_rubric, vf.Rubric(funcs=[judge_response], weights=[1])])

    return ReactAgentEnv(dataset=train_dataset, eval_dataset=eval_dataset, system_prompt=system_prompt, rubric=rubric_group, **kwargs)

