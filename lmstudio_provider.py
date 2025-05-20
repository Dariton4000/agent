import lmstudio as lms
from pick import pick
import asyncio
from lmstudio.tools.functions_def import ToolFunctionDef
from llm_provider_interface import LLMProviderInterface
from lmstudio.model_inference.chat import Chat # Potentially Chat from lmstudio

def _print_fragment(fragment: str):
    """Default on_prediction_fragment handler. Prints the fragment to stdout."""
    print(fragment, end="", flush=True)

class LMStudioProvider(LLMProviderInterface):
    """
    LLMProviderInterface implementation for LM Studio.
    """

    def __init__(self):
        self.client = lms.Client()
        self.model = None

    def load_model(self, model_identifier: str = None, **kwargs) -> None:
        if self.model is not None:
            print(f"Model {self.model.get_info().identifier} is already loaded.")
            # Optionally, ask if the user wants to load a different model
            # and unload the current one. For now, just return.
            return

        available_models = self.client.llm.get_downloaded_models()
        if not available_models:
            print("No models downloaded. Please download a model in LM Studio first.")
            return

        selected_model_info = None
        if model_identifier:
            for model_info in available_models:
                if model_info.identifier == model_identifier:
                    selected_model_info = model_info
                    break
            if not selected_model_info:
                print(f"Model identifier '{model_identifier}' not found among downloaded models.")

        if not selected_model_info:
            model_options = [f"{m.identifier} ({m.architecture})" for m in available_models]
            option, index = pick(model_options, "Select a model to load:")
            selected_model_info = available_models[index]
            model_identifier = selected_model_info.identifier
            print(f"Selected model: {model_identifier}")

        try:
            print(f"LM Studio: Loading model {model_identifier}...")
            # Pass kwargs to load_new_instance if any are relevant
            self.model = self.client.llm.load_new_instance(selected_model_info.identifier, **kwargs)
            # Check if model loading actually succeeded by trying to get info
            if self.model:
                model_info_check = self.model.get_info() # This might throw if model is invalid
                print(f"LM Studio: Model {model_info_check.identifier} loaded successfully.")
                print(f"  Trained for Tool Use: {model_info_check.trained_for_tool_use}")
                print(f"  Context Length: {self.model.get_context_length()}") # Assuming this is safe if get_info() worked
            else: # Should not happen if load_new_instance raises on failure
                print(f"LM Studio: Error loading model {model_identifier}: load_new_instance returned None.")
                self.model = None # Ensure it's None
        except lms.errors.LMStudioSDKError as lms_sdk_error: # More specific LM Studio error
            print(f"LM Studio: SDK Error loading model {model_identifier}: {lms_sdk_error}")
            self.model = None
        except Exception as e: # Catch other potential exceptions
            print(f"LM Studio: Error loading model {model_identifier}: {e}")
            self.model = None

    def get_model_info(self) -> dict:
        if not self.model:
            return {"status": "No model loaded. Call load_model() first.", "provider": "lmstudio"}

        try:
            info = self.model.get_info()
        except Exception as e:
            return {"status": f"Error retrieving model info: {e}", "provider": "lmstudio"}
        return {
            "identifier": info.identifier,
            "trained_for_tool_use": info.trained_for_tool_use,
            "context_length": self.model.get_context_length(),
            # Add other relevant info from info object if needed
            "architecture": info.architecture,
            "status": info.status,
        }

    def chat_completion(self, messages: list, tools: list = None, **kwargs) -> str:
        if not self.model:
            # raise ValueError("No model loaded. Call load_model() first.")
            return "Error: No model loaded. Call load_model() first."

        # Convert messages to lms.Chat object
        # Assuming messages are in the format: {"role": "user/assistant/system", "content": "..."}
        # System message handling might need specific logic based on how lms.Chat expects it.
        # For now, let's assume the first message can be a system prompt if role is 'system'.
        
        chat_session = Chat()
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                if role.lower() == "system" and not chat_session.system_prompt:
                    chat_session.system_prompt = content
                elif role.lower() == "user":
                    chat_session.add_user_message(content)
                elif role.lower() == "assistant":
                    # This might need adjustment if lms.Chat handles assistant messages differently
                    # during setup vs. during interaction.
                    # For now, assuming we are setting up a chat history.
                    chat_session.add_assistant_message(content)
            else:
                print(f"Warning: Skipping invalid message object: {msg}")
        
        # Handle on_prediction_fragment callback
        # For simplicity, collecting the full response.
        # A more advanced implementation might pass a callback or use asyncio.
        full_response_fragments = []
        def collect_fragment(fragment: str):
            full_response_fragments.append(fragment)
            _print_fragment(fragment) # Also print to console like original

        on_fragment_callback = kwargs.pop("on_prediction_fragment", collect_fragment)
        
        # Ensure chat fits in context (simplified check)
        # A more robust check would use self.tokenize and self.get_context_length()
        # This is a placeholder for the `does_chat_fit_in_context` logic
        prompt_string = self.model.apply_prompt_template(chat_session)
        tokens = self.model.tokenize(prompt_string)
        if len(tokens) > self.model.get_context_length():
            print("Warning: The chat may exceed the model's context length.")
            # Handle context overflow (e.g., truncate, summarize, or error)
            # For now, just printing a warning.

        try:
            print("\nAssistant: ", end="", flush=True) # Main.py now handles this prefix
            self.model.act(
                chat_session,
                tools=tools, 
                on_message=lambda msg: chat_session.add_message(msg),
                on_prediction_fragment=on_fragment_callback,
                **kwargs 
            )
            # print() # Main.py handles newline after response
            
            final_response = "".join(full_response_fragments)
            
            # If the model itself had an error during act() and didn't produce a message,
            # chat_session might not have a new assistant message.
            # The final_response from fragments is the primary source of content.
            # If there was a tool call, the final_response might be the result of the tool call
            # if the LLM decides to output that directly. Or it could be a summary.

            return final_response
        except lms.errors.LMStudioSDKError as lms_sdk_error:
             print(f"\nLM Studio: SDK Error during chat completion: {lms_sdk_error}")
             # Depending on severity, you might want to raise or return an error message
             return f"Error: LM Studio SDK error during chat: {lms_sdk_error}"
        except Exception as e:
            print(f"\nLM Studio: Error during chat completion: {e}")
            return f"Error: {e}"

    def apply_prompt_template(self, messages: list) -> str:
        if not self.model:
            return "Error: No model loaded. Call load_model() first."
        
        try:
            chat_session = Chat()
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role and content:
                    if role.lower() == "system" and not chat_session.system_prompt:
                        chat_session.system_prompt = content
                    elif role.lower() == "user":
                        chat_session.add_user_message(content)
                    elif role.lower() == "assistant":
                        chat_session.add_assistant_message(content)
            
            return self.model.apply_prompt_template(chat_session)
        except Exception as e:
            print(f"LM Studio: Error applying prompt template: {e}")
            # Return a basic concatenation as a fallback, or an error string
            return "Error applying prompt template. " + " ".join(m.get("content", "") for m in messages)


    def tokenize(self, text: str) -> list:
        if not self.model:
            print("LM Studio: Cannot tokenize, no model loaded.")
            return [] 
        try:
            return self.model.tokenize(text)
        except Exception as e:
            print(f"LM Studio: Error during tokenization: {e}")
            return []

    def get_context_length(self) -> int:
        if not self.model:
            print("LM Studio: Cannot get context length, no model loaded.")
            return 0 
        try:
            return self.model.get_context_length()
        except Exception as e:
            print(f"LM Studio: Error getting context length: {e}")
            return 0
    # Helper methods (if any specific to this provider beyond _print_fragment)
    # For example, a more robust does_chat_fit_in_context could be here:
    # Note: _does_chat_fit_in_context is not currently used by main.py's check_context_fit
    # as check_context_fit implements its own logic using provider's tokenize and get_context_length.
    def _does_chat_fit_in_context(self, chat_session: Chat, max_tokens_to_generate: int = 0) -> bool: # Unused
        if not self.model:
            print("LM Studio: Model not loaded, cannot check context fit.")
            return False
        
        try:
            # This is a simplified version. LM Studio's internal check is more accurate.
            # We might rely on the model.act() to handle this or provide a more precise check.
            prompt_string = self.model.apply_prompt_template(chat_session)
            tokens = self.model.tokenize(prompt_string)
            context_length = self.model.get_context_length()
            remaining_context = context_length - len(tokens)
            
            if remaining_context < max_tokens_to_generate: # Or some buffer
                print(f"Warning: Chat context ({len(tokens)} tokens) + generation ({max_tokens_to_generate} tokens) may exceed context length ({context_length}). Remaining: {remaining_context}")
                return False
            return True
        except Exception as e:
            print(f"LM Studio: Error in _does_chat_fit_in_context: {e}")
            return False # Assume it doesn't fit on error

# Example Usage (for testing purposes, typically not part of the provider file)
if __name__ == "__main__":
    # This part would require LM Studio to be running and a model downloaded.
    # It's also assuming synchronous execution for simplicity here.
    # Real usage would be through the main application logic.

    print("Attempting to initialize LMStudioProvider...")
    provider = LMStudioProvider()
    
    # Test load_model
    # You might need to run LM Studio and download a model first.
    # Replace "model-identifier" with an actual identifier if you want to test specific loading.
    # provider.load_model() # Interactive selection
    # Example: provider.load_model(model_identifier="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf") 
    # For this example, we'll try interactive loading if no model is pre-loaded by other tests.
    if not provider.model:
        print("No model pre-loaded, attempting interactive load for testing...")
        provider.load_model()


    if not provider.model:
        print("LM Studio: Model could not be loaded. Exiting test.")
    else:
        print("\n--- Testing get_model_info ---")
        info = provider.get_model_info()
        print(f"Model Info: {info}")

        print("\n--- Testing tokenize ---")
        tokens = provider.tokenize("Hello, world!")
        print(f"Tokens for 'Hello, world!': {tokens} (Count: {len(tokens)})")

        print("\n--- Testing get_context_length ---")
        context_length = provider.get_context_length()
        print(f"Context Length: {context_length}")

        print("\n--- Testing apply_prompt_template ---")
        sample_messages_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        prompt = provider.apply_prompt_template(sample_messages_template)
        print(f"Applied prompt template:\n{prompt}")

        print("\n--- Testing chat_completion ---")
        sample_messages_chat = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        # To test tools, you would define ToolFunctionDef objects
        # example_tool = ToolFunctionDef(
        #     name="get_weather",
        #     description="Gets the current weather for a location.",
        #     params_jsonschema={
        #         "type": "object",
        #         "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}},
        #         "required": ["location"],
        #     }
        # )
        # response = provider.chat_completion(sample_messages_chat, tools=[example_tool])
        
        response = provider.chat_completion(sample_messages_chat)
        print(f"\nFinal Chat Response from provider: {response}")

        print("\n--- Testing chat_completion with a slightly longer conversation ---")
        longer_convo = [
            {"role": "system", "content": "You are a helpful AI."},
            {"role": "user", "content": "My name is Bob."},
            {"role": "assistant", "content": "Nice to meet you Bob!"},
            {"role": "user", "content": "What is my name?"}
        ]
        response_name = provider.chat_completion(longer_convo)
        print(f"\nFinal Chat Response to 'What is my name?': {response_name}")

    print("\nLMStudioProvider tests finished.")
