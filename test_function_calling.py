#!/usr/bin/env python3
"""
Test script to check if gpt-nano models support function calling.
"""
from crewai.llm import LLM


def test_nano_function_calling():
    """Test if gpt nano models support function calling."""
    
    models_to_test = [
        "gpt-4.1-nano-2025-04-14",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-4o"
    ]
    
    for model in models_to_test:
        try:
            llm = LLM(model=model)
            supports_fc = llm.supports_function_calling()
            print(f"{model}: Function calling supported = {supports_fc}")
        except Exception as e:
            print(f"{model}: Error checking function calling - {e}")


if __name__ == "__main__":
    test_nano_function_calling()