"""
LLM Generator:
Generates grounded medical answers using a language model and retrieved context.
Supports FLAN-T5, Mistral, and OpenAI-compatible APIs.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

ANSWER_PROMPT_TEMPLATE = """You are a knowledgeable and careful medical assistant.
Use ONLY the provided context to answer the question.
If the context does not contain enough information, say "I don't have enough information to answer this question reliably."
Do NOT add information beyond what is in the context.
Always be concise, clear, and accurate.

Context:
{context}

Question: {question}

Answer:"""


class LLMGenerator:
    """
    Wraps a generative LLM for answer generation from retrieved context.

    Supported backends:
      - HuggingFace (FLAN-T5, Mistral)
      - OpenAI API (gpt-3.5-turbo, gpt-4)
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        use_openai: bool = False,
        openai_model: str = "gpt-3.5-turbo",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_openai = use_openai
        self.openai_model = openai_model
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        if self.use_openai:
            return  # OpenAI uses API calls, no local load needed
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

        logger.info(f"Loading LLM: {self.model_name}")
        if "t5" in self.model_name.lower():
            self._pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        else:
            # Causal LM (Mistral, LLaMA, etc.)
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
            )

    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer grounded in the retrieved context.

        Args:
            question: The user's medical question.
            context:  Retrieved and concatenated document context.

        Returns:
            Generated answer as a string.
        """
        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)

        if self.use_openai:
            return self._generate_openai(prompt)
        return self._generate_hf(prompt)

    def _generate_hf(self, prompt: str) -> str:
        self._load_pipeline()
        result = self._pipeline(prompt)
        if isinstance(result, list) and len(result) > 0:
            output = result[0]
            if "generated_text" in output:
                text = output["generated_text"]
                # For causal LM, strip the prompt prefix
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()
                return text
        return "Unable to generate an answer."

    def _generate_openai(self, prompt: str) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Unable to generate an answer due to API error."
