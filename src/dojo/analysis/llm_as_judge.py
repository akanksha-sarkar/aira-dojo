import os
import json
from typing import Any, Dict, Optional
from openai import OpenAI

## Average cost: ~17000 input tokens, ~250 completion tokens -> $0.02 per program
SYSTEM_PROMPT = """
You are an expert in data-efficient machine learning and semi-supervised learning methods.

Given a program, extract concise and accurate descriptors of:
1. Models used: backbone architectures (e.g., ResNet50, ViT, CLIP, DINOv2)
2. Methods used: training strategies, learning paradigms, regularization schemes,
   pseudo-labeling approaches, consistency objectives, augmentation schemes,
   parameter-efficient finetuning methods, or other notable learning or unsupervised methods.
3. Add a one word binary label on whether the program uses any training of models or adapter heads.

Return ONLY valid JSON in this format:

{
    "models": ["model1", "model2"],
    "methods": ["method1", "method2"],
    "uses training": "yes" or "no",
    "summary": "A concise summary of the program's purpose and main features (1-5 sentences)."
}

Guidelines:
- Include backbone or model family names when identifiable.
- Keep the descriptions concise (a few words) but accurate.
- Include a descriptive summary of the program's purpose and main features (1-5 sentences).
- Focus on the features and methods used.
"""



class LLMJudge:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt

    def judge_program(self, program: str, program_id: Optional[str] = None) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Program:\n\n{program}\n\nExtract the models and methods.",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content
        parsed_output = self.parse_output(raw_output)

        usage_dict = self.extract_usage(response)

        result = {
            "program_id": program_id,
            "model_name": self.model,
            "parsed_output": parsed_output,
            "raw_output": raw_output,
            "usage": usage_dict,
        }
        return result

    def parse_output(self, output: str) -> Dict[str, Any]:
        try:
            data = json.loads(output)
            return {
                "models": data.get("models", []),
                "methods": data.get("methods", []),
                "uses training": data.get("uses training", "no"),
                "summary": data.get("summary", ""),
            }
        except json.JSONDecodeError:
            return {
                "models": [],
                "methods": [],
                "uses training": "no",
                "summary": "",
                "parse_error": True,
                "raw": output,
            }

    def extract_usage(self, response) -> Optional[Dict[str, Any]]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        # Works across SDK variants a bit more safely than assuming .dict()
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

        # Include extra fields if present in your SDK/model response
        cached_tokens = getattr(usage, "prompt_tokens_details", None)
        if cached_tokens is not None:
            usage_dict["prompt_tokens_details"] = (
                cached_tokens.model_dump()
                if hasattr(cached_tokens, "model_dump")
                else str(cached_tokens)
            )

        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details is not None:
            usage_dict["completion_tokens_details"] = (
                completion_details.model_dump()
                if hasattr(completion_details, "model_dump")
                else str(completion_details)
            )

        return usage_dict

def load_program(program_path: str) -> str:
    with open(program_path, "r") as f:
        return f.read()

def judge_program(program: str, program_id: Optional[str] = None, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    judge = LLMJudge(model=model)
    return judge.judge_program(program, program_id)

if __name__ == "__main__":
    program = load_program("src/dojo/analysis_utils/example_program.py")
    result = judge_program(program, program_id="program_001", model="gpt-4o-mini")
    print(result)