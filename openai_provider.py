import openai

def load_openai_model(api_key, model_name):
    openai.api_key = api_key
    return model_name

def interact_with_openai_model(model_name, prompt):
    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()
