from abc import ABC, abstractmethod

class LLMProviderInterface(ABC):
    """
    Interface for LLM providers.
    """

    @abstractmethod
    def load_model(self, model_identifier: str, **kwargs) -> None:
        """
        Load or set up a model.

        Args:
            model_identifier: Local model name or API model name.
            **kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model.

        Returns:
            A dictionary containing model information (e.g., context length,
            supported features).
        """
        pass

    @abstractmethod
    def chat_completion(self, messages: list, tools: list = None, **kwargs) -> str:
        """
        Get a chat completion from the model.

        Args:
            messages: A list of message objects.
            tools: A list of tool definitions.
            **kwargs: Additional keyword arguments.

        Returns:
            The chat completion response from the model.
        """
        pass

    @abstractmethod
    def apply_prompt_template(self, messages: list) -> str:
        """
        Apply the provider-specific prompt template.

        Args:
            messages: A list of message objects.

        Returns:
            The prompt string with the template applied.
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list:
        """
        Tokenize text according to the model's tokenizer.

        Args:
            text: The text to tokenize.

        Returns:
            A list of tokens.
        """
        pass

    @abstractmethod
    def get_context_length(self) -> int:
        """
        Get the context length of the model.

        Returns:
            The context length of the model.
        """
        pass
