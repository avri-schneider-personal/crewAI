#!/usr/bin/env python3
"""
Test script to diagnose the gpt-nano tool execution termination issue.
"""
import os
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool


class SimpleTestTool(BaseTool):
    name: str = "simple_test_tool"
    description: str = "A simple test tool that returns a greeting"

    def _run(self, name: str = "World") -> str:
        return f"Hello, {name}!"


def test_gpt_nano_tool_execution():
    """Test tool execution with gpt nano model."""
    
    # Create agent with gpt-4.1-nano model
    agent = Agent(
        role="Test Agent",
        goal="Test tool execution",
        backstory="A test agent to verify tool execution works",
        verbose=True,
        tools=[SimpleTestTool()],
        llm="gpt-4.1-nano-2025-04-14"  # Using the nano model from tests
    )
    
    # Create a simple task that should use the tool
    task = Task(
        description="Use the simple_test_tool to greet 'CrewAI'. Make sure to actually call the tool, don't just give a final answer without using the tool.",
        expected_output="A greeting message from the tool",
        agent=agent
    )
    
    # Execute the task
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    
    print("=== RESULT ===")
    print(f"Raw result: {result.raw}")
    print(f"Result type: {type(result)}")
    
    # Check if tool was actually executed (the output should contain tool results)
    if "Hello, CrewAI!" in result.raw:
        print("SUCCESS: Tool was executed properly")
    else:
        print("ISSUE: Tool was not executed - likely terminated with Final Answer")
    
    return result


if __name__ == "__main__":
    # Set up environment - you may need to set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-key-here"
    
    print("Testing GPT Nano tool execution...")
    test_gpt_nano_tool_execution()