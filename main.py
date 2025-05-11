import lmstudio as lms
import time
import pick


if lms.list_loaded_models() == []:
    print("No models loaded")
    title = 'Please choose a model to load:'
    options = [model.model_key for model in lms.list_downloaded_models("llm")]
    option, index = pick.pick(options, title)
    with lms.Client() as client:
        model = client.llm.load_new_instance(options[index])
else:
    with lms.Client() as client:
        model = client.llm.model()
        if model.get_info().trained_for_tool_use == False:
            print("Model is not trained for tool use."
                  "Please be careful when using it.")
        else:
            print(model.get_info().identifier, "is already loaded.")
            if input("Unload model? (y/N)") == "y":
                with lms.Client() as client:
                    model = client.llm.model()
                    model.unload()


