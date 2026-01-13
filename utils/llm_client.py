from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage
import json

class LLMClient:
    def __init__(self, model="gpt-4", api_key=None):
        self.model_name = model
        if "gpt" in model.lower():
            self.client = ChatOpenAI(model_name=model, openai_api_key=api_key, temperature=0)
        elif "claude" in model.lower():
            self.client = ChatAnthropic(model=model, anthropic_api_key=api_key, temperature=0)
        else:
            raise ValueError(f"Unsupported model: {model}")

    def score_response(self, text, response, language="en", rubric=None):
        """
        Send text + response + rubric to LLM and return JSON with score + feedback.
        """
        rubric_text = rubric or "Score the response 0-1 for accuracy, relevance, clarity, and cultural correctness."
        prompt = f"""
Text: {text}
Response: {response}
Language: {language}
Instructions: {rubric_text}

Return JSON: {{"score": 0-1, "feedback": "..."}} 
"""
        reply = self.client([HumanMessage(content=prompt)])
        try:
            return json.loads(reply.content)
        except:
            return {"score": None, "feedback": reply.content}
