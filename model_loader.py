import lmstudio as lms
import pick

def load_model():
    """
    Loads a language model, prompting the user to select or unload models as needed.
    
    If no model is loaded, prompts the user to select a downloaded LLM model and loads it with a context length of 32,768 tokens. If a model is already loaded, checks if it is trained for tool use and offers the option to unload and select a different model.
    
    Returns:
        The loaded language model instance.
    """
    if lms.list_loaded_models() == []:
        print("No models loaded")
        title = 'Please choose a model to load:'
        options = [model.model_key for model in lms.list_downloaded_models("llm")]
        option, index = pick.pick(options, title)
        with lms.Client() as client:
            
            model = client.llm.load_new_instance(options[index], config={
                "contextLength": 32768,
            })
            print("Model Loaded")
            return model
    else:
        with lms.Client() as client:
            model = client.llm.model()
            if model.get_info().trained_for_tool_use == False:
                print("Model is not trained for tool use. "
                      "Please be careful when using it.")
                if input("Unload model? (y/N)") == "y":
                    model.unload()
                    return load_model()
                else:
                    return model
            else:
                print(model.get_info().identifier, "is already loaded.")
                if input("Unload model? (y/N)") == "y":
                    model.unload()
                    return load_model()
                else:
                    return model

def does_chat_fit_in_context(model: lms.LLM, chat: lms.Chat):
    # Convert the conversation to a string using the prompt template.
    """
    Calculates the ratio of the model's context length to the token count of a chat conversation.
    
    Args:
        model: The language model instance to evaluate against.
        chat: The chat conversation to assess.
    
    Returns:
        The ratio of the model's context window size to the number of tokens in the chat.
    """
    formatted = model.apply_prompt_template(chat)
    # Count the number of tokens in the string.
    token_count = len(model.tokenize(formatted))
    # Get the current loaded context length of the model
    context_length = model.get_context_length()
    return context_length / token_count
