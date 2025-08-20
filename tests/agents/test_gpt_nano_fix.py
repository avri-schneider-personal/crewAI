"""
Additional test cases for the gpt-nano tool execution fix.
These tests verify that when both Action and Final Answer patterns are present,
the parser correctly prioritizes tool execution over final answers.
"""
from crewai.agents.parser import CrewAgentParser, AgentAction, AgentFinish


class MockAgent:
    def increment_formatting_errors(self):
        pass


def test_action_prioritized_over_final_answer():
    """Test that action execution is prioritized when both patterns are present."""
    parser = CrewAgentParser(MockAgent())
    
    text = """Thought: I need to use a tool to get information.
Action: simple_test_tool
Action Input: {"name": "CrewAI"}

Final Answer: Based on the tool results, here's my answer."""
    
    result = parser.parse(text)
    
    # Should prioritize action execution over final answer
    assert isinstance(result, AgentAction)
    assert result.tool == "simple_test_tool"
    assert "CrewAI" in result.tool_input


def test_nano_model_typical_response_pattern():
    """Test parsing a typical gpt-nano response that includes both patterns."""
    parser = CrewAgentParser(MockAgent())
    
    text = """Thought: I need to greet the user using the tool.
        
Action: simple_test_tool
Action Input: {"name": "User"}

The tool will provide a greeting.

Final Answer: I'll use the simple test tool to provide a proper greeting."""
    
    result = parser.parse(text)
    
    # Should execute the tool, not terminate with final answer
    assert isinstance(result, AgentAction)
    assert result.tool == "simple_test_tool"
    assert "User" in result.tool_input


def test_complex_response_with_multiple_sections():
    """Test complex response that might come from nano models."""
    parser = CrewAgentParser(MockAgent())
    
    text = """Thought: I need to analyze this step by step.

I'll start by using the search tool to gather information.

Action: search_tool  
Action Input: {"query": "important data", "type": "comprehensive"}

After I get the results, I'll process them and provide a final answer.

Final Answer: Based on my analysis using the search tool, here are the findings."""
    
    result = parser.parse(text)
    
    # Should execute the tool despite the final answer section
    assert isinstance(result, AgentAction)
    assert result.tool == "search_tool"
    assert "important data" in result.tool_input


def test_final_answer_before_action_edge_case():
    """Test case where Final Answer appears before Action (edge case)."""
    parser = CrewAgentParser(MockAgent())
    
    text = """Thought: I'm thinking about this.

Final Answer: Wait, I need to use a tool first.

Action: simple_test_tool
Action Input: {"name": "Test"}"""
    
    result = parser.parse(text)
    
    # Should still prioritize action execution
    assert isinstance(result, AgentAction)
    assert result.tool == "simple_test_tool"


def test_backward_compatibility_action_only():
    """Ensure existing action-only parsing still works."""
    parser = CrewAgentParser(MockAgent())
    
    text = """Thought: I need to use a tool.
Action: simple_test_tool
Action Input: {"name": "CrewAI"}"""
    
    result = parser.parse(text)
    
    assert isinstance(result, AgentAction)
    assert result.tool == "simple_test_tool"


def test_backward_compatibility_final_answer_only():
    """Ensure existing final-answer-only parsing still works."""
    parser = CrewAgentParser(MockAgent())
    
    text = """Thought: I have all the information I need.
Final Answer: The result is 42."""
    
    result = parser.parse(text)
    
    assert isinstance(result, AgentFinish)
    assert result.output == "The result is 42."