from typing import Optional, Dict, Any, Union
import json
from tenacity import retry,  stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from ..utils.logger import logger
from .config import config

class LLMClient:
    """
    Enterprise-grade LLM Client supporting multiple providers with automatic retries 
    and robust error handling.
    """
    
    def __init__(self, model: str = config.DEFAULT_MODEL, api_key: Optional[str] = None):
        self.model_name = model
        self.api_key = api_key
        self._client = self._initialize_client()
        logger.info(f"Initialized LLMClient with model: {self.model_name}")

    def _initialize_client(self):
        """Initializes the appropriate LangChain chat model."""
        try:
            if "gpt" in self.model_name.lower():
                key = self.api_key or config.OPENAI_API_KEY
                if not key:
                    raise ValueError("OpenAI API Key not found.")
                return ChatOpenAI(model_name=self.model_name, openai_api_key=key, temperature=0)
            
            elif "claude" in self.model_name.lower():
                key = self.api_key or config.ANTHROPIC_API_KEY
                if not key:
                    raise ValueError("Anthropic API Key not found.")
                return ChatAnthropic(model=self.model_name, anthropic_api_key=key, temperature=0)
            
            else:
                raise ValueError(f"Unsupported model family: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generates text with automatic retries.
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            logger.debug(f"Sending request to {self.model_name}...")
            response = self._client.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            raise

    def score_response(
        self, 
        text: str, 
        response: str, 
        language: str = "en", 
        rubric: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scores a response against a rubric, returning structured JSON.
        """
        default_rubric = "Score the response 0-1 for accuracy, relevance, clarity, and cultural correctness."
        rubric_text = rubric or default_rubric
        
        prompt = f"""
        You are an AI Quality Assurance Evaluator.
        
        Task: Evaluate the following response.
        Input Text ({language}): {text}
        Response to Evaluate: {response}
        Evaluation Criteria: {rubric_text}
        
        Output Requirements: 
        Return ONLY valid JSON with the following keys:
        - "score": (float 0.0 to 1.0)
        - "feedback": (concise string explanation)
        - "reasoning": (detailed bullet points if needed)
        """
        
        try:
            raw_output = self.generate_text(prompt)
            # Remove any markdown formatting if present
            cleaned_output = raw_output.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_output)
        except json.JSONDecodeError:
            logger.warning("LLM returned malformed JSON. Fallback to raw text.")
            return {"score": 0.0, "feedback": "Error parsing LLM output", "raw_output": raw_output}
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return {"score": 0.0, "feedback": f"System Error: {str(e)}"}
