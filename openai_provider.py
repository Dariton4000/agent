import openai

def load_openai_model(api_key, model_name):
    """
    Sets the OpenAI API key and returns the specified model name.
    
    Args:
        api_key: The API key to authenticate with the OpenAI service.
        model_name: The name of the OpenAI model to use.
    
    Returns:
        The provided model name.
    """
    openai.api_key = api_key
    return model_name

def interact_with_openai_model(model_name, prompt):
    """
    Sends a prompt to an OpenAI model and returns the generated completion.
    
    Args:
        model_name: The identifier of the OpenAI model to use.
        prompt: The input text to send to the model.
    
    Returns:
        The generated text completion from the model, with leading and trailing whitespace removed.
    """
    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()
