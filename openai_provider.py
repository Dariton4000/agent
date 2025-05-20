import openai
import os
import json
from llm_provider_interface import LLMProviderInterface
# from lmstudio.tools.functions_def import ToolFunctionDef # For type hinting if using directly

# Placeholder for actual tool execution logic
def _execute_tool(tool_name: str, tool_args: dict, available_tools: list):
    print(f"OpenAI: Attempting to execute tool: {tool_name} with args: {tool_args}")
    for tool_def in available_tools:
        if hasattr(tool_def, 'name') and tool_def.name == tool_name:
            if hasattr(tool_def, 'implementation'):
                try:
                    result = tool_def.implementation(**tool_args)
                    return result
                except Exception as e:
                    print(f"OpenAI Error: Exception during execution of tool '{tool_name}': {e}")
                    return f"Error executing tool {tool_name}: {e}"
            else:
                return f"Tool '{tool_name}' has no implementation attribute."
    return f"Tool '{tool_name}' not found or not executable."

class OpenAIProvider(LLMProviderInterface):
    MODEL_CONTEXT_LENGTHS = {
        "gpt-4-turbo-preview": 128000, "gpt-4-0125-preview": 128000, "gpt-4-1106-preview": 128000,
        "gpt-4-vision-preview": 128000, "gpt-4": 8192, "gpt-4-0613": 8192,
        "gpt-4-32k": 32768, "gpt-4-32k-0613": 32768, "gpt-3.5-turbo-0125": 16385,
        "gpt-3.5-turbo": 16385, "gpt-3.5-turbo-1106": 16385, "gpt-3.5-turbo-instruct": 4096,
    }

    def __init__(self, api_key: str = None, base_url: str = None):
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OpenAI API key not provided and not found in OPENAI_API_KEY environment variable. Please set OPENAI_API_KEY or provide the key directly.")
        try:
            self.client = openai.OpenAI(api_key=resolved_api_key, base_url=base_url)
            self.model_identifier = None
            print("OpenAIProvider initialized successfully.")
            if base_url:
                print(f"Using custom OpenAI base URL: {base_url}")
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")

    def load_model(self, model_identifier: str, **kwargs) -> None:
        if not model_identifier:
            raise ValueError("OpenAI model identifier cannot be empty.")
        self.model_identifier = model_identifier
        print(f"OpenAI model identifier set to: {self.model_identifier}.")
        try:
            print(f"Verifying model '{self.model_identifier}' with OpenAI API...")
            model_data = self.client.models.retrieve(self.model_identifier)
            print(f"Successfully verified model '{self.model_identifier}' with OpenAI API. ID: {model_data.id}")
        except openai.NotFoundError:
            print(f"OpenAI Error: Model '{self.model_identifier}' not found.")
            self.model_identifier = None
            raise ValueError(f"Model '{self.model_identifier}' not found. Please check the model name and ensure it's available for your API key.")
        except openai.AuthenticationError as auth_err:
            print(f"OpenAI Authentication Error while verifying model: {auth_err}")
            self.model_identifier = None
            raise ValueError(f"OpenAI Authentication Error: {auth_err}. Check your API key and organization.")
        except openai.APIConnectionError as conn_err:
            print(f"OpenAI API Connection Error while verifying model: {conn_err}")
            self.model_identifier = None
            raise ValueError(f"OpenAI API Connection Error: {conn_err}. Check your network connection and the API endpoint.")
        except openai.RateLimitError as rate_err:
            print(f"OpenAI Rate Limit Error while verifying model: {rate_err}")
            self.model_identifier = None
            raise ValueError(f"OpenAI Rate Limit Error: {rate_err}. Please check your rate limits.")
        except openai.APIError as api_err:
            print(f"OpenAI API Error while verifying model '{self.model_identifier}': {api_err}")
            self.model_identifier = None
            raise ValueError(f"OpenAI API Error verifying model: {api_err}")
        except Exception as e:
            print(f"An unexpected error occurred while trying to verify model '{self.model_identifier}': {e}")
            self.model_identifier = None
            raise ValueError(f"Unexpected error verifying model: {e}")

    def get_model_info(self) -> dict:
        if not self.model_identifier:
            return {"status": "No model loaded/selected. Call load_model() first.", "provider": "openai", "identifier": None, "context_length": 0}
        context_length = self.get_context_length()
        return {"identifier": self.model_identifier, "context_length": context_length if context_length != -1 else "Context length unknown or varies", "provider": "openai", "supported_features": ["chat_completion", "tool_use"]}

    def _convert_tools_to_openai_format(self, tools: list) -> list:
        if not tools: return None
        openai_tools = []
        for tool_def in tools:
            if not all(hasattr(tool_def, attr) for attr in ['name', 'description', 'params_jsonschema']):
                print(f"OpenAI Warning: Skipping tool {getattr(tool_def, 'name', 'Unnamed tool')} due to missing attributes for OpenAI conversion.")
                continue
            openai_tools.append({"type": "function", "function": {"name": tool_def.name, "description": tool_def.description, "parameters": tool_def.params_jsonschema}})
        return openai_tools if openai_tools else None

    def chat_completion(self, messages: list, tools: list = None, **kwargs) -> str:
        if not self.model_identifier:
            print("OpenAI Error: No model identifier set. Cannot perform chat completion.")
            return "Error: No model identifier set for OpenAI. Please load a model first."
        
        formatted_messages = []
        for msg in messages:
            if "role" in msg and "content" in msg: formatted_messages.append({"role": msg["role"], "content": msg["content"]})
            elif "role" in msg and msg["role"] == "tool" and "tool_call_id" in msg and "content" in msg: formatted_messages.append({"role": "tool", "tool_call_id": msg["tool_call_id"], "content": msg["content"]})
            else: print(f"OpenAI Warning: Skipping malformed message: {msg}")
        if not formatted_messages: return "Error: No valid messages to send to OpenAI."

        openai_tools_formatted = self._convert_tools_to_openai_format(tools)
        tool_choice = kwargs.get("tool_choice", "auto" if openai_tools_formatted else None)

        try:
            print(f"OpenAI: Sending request to model: {self.model_identifier} with {len(formatted_messages)} messages.")
            if openai_tools_formatted: print(f"OpenAI: Tools provided: {[t['function']['name'] for t in openai_tools_formatted]}")

            response = self.client.chat.completions.create(
                model=self.model_identifier, messages=formatted_messages,
                tools=openai_tools_formatted, tool_choice=tool_choice,
                **{k: v for k, v in kwargs.items() if k not in ["tool_choice"]}
            )
            response_message = response.choices[0].message

            if response_message.tool_calls:
                print(f"OpenAI: Response includes tool calls: {response_message.tool_calls}")
                # Important: Append the assistant message that contains the tool_calls to the history
                # The 'openai' library response_message is already a dict-like object
                formatted_messages.append(dict(response_message))


                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args_json = tool_call.function.arguments
                    try:
                        function_args = json.loads(function_args_json)
                    except json.JSONDecodeError as e:
                        print(f"OpenAI Error: Could not decode JSON arguments for tool {function_name}: {e}. Arguments: '{function_args_json}'")
                        error_content = f"Error: Invalid JSON arguments for tool {function_name}. Details: {e}"
                        formatted_messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": error_content})
                        continue
                    
                    print(f"OpenAI: Executing tool: {function_name} with args: {function_args}")
                    tool_result = _execute_tool(function_name, function_args, tools or [])
                    result_snippet = str(tool_result)[:200] + "..." if len(str(tool_result)) > 200 else str(tool_result)
                    print(f"OpenAI: Tool {function_name} executed. Result snippet: {result_snippet}")
                    formatted_messages.append({"tool_call_id": tool_call.id, "role": "tool", "content": str(tool_result)})
                
                print(f"OpenAI: Re-sending request to model {self.model_identifier} with tool responses ({len(formatted_messages)} messages total).")
                final_response_obj = self.client.chat.completions.create(
                    model=self.model_identifier, messages=formatted_messages,
                    **{k: v for k, v in kwargs.items() if k not in ["tool_choice", "tools"]}
                )
                return final_response_obj.choices[0].message.content
            return response_message.content
        except openai.AuthenticationError as e:
            print(f"OpenAI Authentication Error: {e}. Check API key/organization.")
            raise # Re-raise for main loop to handle potentially
        except openai.RateLimitError as e:
            print(f"OpenAI Rate Limit Error: {e}. Check usage/rate limits.")
            return f"Error: OpenAI Rate Limit Exceeded - {e}"
        except openai.APIConnectionError as e:
            print(f"OpenAI API Connection Error: {e}. Check network/OpenAI status.")
            return f"Error: OpenAI Connection Problem - {e}"
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            return f"Error: OpenAI API Error - {e}"
        except Exception as e:
            print(f"An unexpected error occurred during OpenAI chat completion: {e}")
            return f"Error: Unexpected error during chat with OpenAI - {e}"

    def apply_prompt_template(self, messages: list) -> str:
        try:
            return json.dumps(messages, indent=2)
        except Exception as e:
            print(f"OpenAI Error serializing messages for prompt template: {e}")
            return str(messages)

    def tokenize(self, text: str) -> list:
        print("OpenAI: Tokenize method is a passthrough. For token counts, use 'tiktoken' library client-side.")
        return []

    def get_context_length(self) -> int:
        if not self.model_identifier:
            print("OpenAI Warning: Model identifier not set. Cannot determine context length. Returning -1.")
            return -1
        length = self.MODEL_CONTEXT_LENGTHS.get(self.model_identifier)
        if length is not None: return length
        for model_prefix in sorted(self.MODEL_CONTEXT_LENGTHS.keys(), key=len, reverse=True):
            if self.model_identifier.startswith(model_prefix):
                inferred_length = self.MODEL_CONTEXT_LENGTHS[model_prefix]
                print(f"OpenAI Warning: Exact context length for '{self.model_identifier}' is unknown. Inferred {inferred_length} from prefix '{model_prefix}'. Add to MODEL_CONTEXT_LENGTHS for precision.")
                return inferred_length
        print(f"OpenAI Warning: Context length for '{self.model_identifier}' is unknown/not inferred. Returning -1. Update MODEL_CONTEXT_LENGTHS.")
        return -1

# Example Usage
if __name__ == "__main__":
    print("--- Testing OpenAIProvider ---")
    try:
        # provider = OpenAIProvider() # Needs OPENAI_API_KEY env var
        # For testing with a local OpenAI-compatible server (e.g., LM Studio):
        provider = OpenAIProvider(base_url="http://localhost:1234/v1", api_key="lm-studio-key") # API key can be anything for local servers
    except ValueError as e:
        print(f"Initialization failed: {e}")
        exit()

    try:
        # Use a model name that your local server (or OpenAI) would recognize
        # For LM Studio, this can be arbitrary if the server doesn't validate it via API key,
        # or should be one of the loaded models if it does.
        model_to_test = "local-model/phi-2-gguf" # Example for a local setup
        print(f"\n--- Testing load_model with '{model_to_test}' ---")
        provider.load_model(model_identifier=model_to_test)
    except ValueError as e:
        print(f"Failed to load model: {e}")
        # Depending on test, might exit or continue with no model
        # For this test, we want to see other methods fail gracefully if model isn't loaded.
    
    print("\n--- Testing get_model_info ---")
    info = provider.get_model_info()
    print(f"Model Info: {info}")

    print("\n--- Testing tokenize ---")
    tokens = provider.tokenize("Hello, world! This is a test.")
    print(f"Tokenize (illustrative): {tokens}")

    print("\n--- Testing apply_prompt_template ---")
    sample_messages_template = [{"role": "system", "content": "Assistant."},{"role": "user", "content": "Hi?"}]
    prompt_str = provider.apply_prompt_template(sample_messages_template)
    print(f"Applied prompt template (JSON):\n{prompt_str}")

    if provider.model_identifier: # Only proceed if model was successfully set
        print("\n--- Testing chat_completion (simple) ---")
        chat_messages_simple = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "What is 1+1?"}]
        response_simple = provider.chat_completion(chat_messages_simple)
        print(f"Chat Completion (Simple) Response: {response_simple}")

        # Tool definition for testing
        class MockToolFunctionDef:
            def __init__(self, name, desc, schema, func): self.name, self.description, self.params_jsonschema, self.implementation = name, desc, schema, func
        def get_weather(location: str, unit: str = "celsius"):
            if "tokyo" in location.lower(): return json.dumps({"temp": "15", "unit": unit})
            return json.dumps({"temp": "unknown"})
        weather_tool = MockToolFunctionDef("get_weather", "Weather.", {"type": "object", "properties": {"location": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}, get_weather)
        
        print("\n--- Testing chat_completion with a tool ---")
        # Model must be tool-capable. For local, this depends on the model loaded in LM Studio.
        chat_messages_tool = [{"role": "user", "content": "Weather in Tokyo?"}]
        response_tool = provider.chat_completion(chat_messages_tool, tools=[weather_tool])
        print(f"Chat Completion (Tool) Response: {response_tool}")
    else:
        print("\nSkipping chat_completion tests as model was not successfully loaded/set.")

    print("\n--- OpenAIProvider tests finished. ---")
