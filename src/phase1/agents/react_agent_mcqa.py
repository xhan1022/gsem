"""ReAct Agent for MCQA trajectory generation."""
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.shared.config import config
from src.shared.logger import logger


REACT_MCQA_SYSTEM_PROMPT = """You are an expert physician solving medical multiple-choice questions with ReAct reasoning.

Use this exact format:
Step 1:
Thought 1: ...
Action 1: ...
Observation 1: ...

...

Step N:
Thought N: Final synthesis
Action N: Select the best option
Observation N: Briefly verify consistency with the question/options
Final Answer: <option letter>

Rules:
1. Use only information in the question/options.
2. Do not invent additional findings.
3. Final Answer must be a single option letter.
"""


class ReActAgentMCQA:
    """ReAct agent for clinical MCQA."""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=config.deepseek.temperature,
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )

    def generate_trajectory(self, case: Dict[str, Any]) -> Dict[str, Any]:
        try:
            case_id = case.get("id", case.get("case_id", "unknown"))
            question = case.get("question", case.get("description", ""))
            options = case.get("options", {})
            options_text = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())]) if options else ""
            human_prompt = f"Question:\n{question}\n\nOptions:\n{options_text}\n\nPlease use ReAct reasoning."

            messages = [
                SystemMessage(content=REACT_MCQA_SYSTEM_PROMPT),
                HumanMessage(content=human_prompt),
            ]

            response = self.llm.invoke(messages)
            trajectory_text = response.content
            steps = self._parse_trajectory(trajectory_text)

            final_answer = ""
            total_steps = 0
            for step in steps:
                if step["type"] == "final_answer":
                    final_answer = step["content"]
                step_num = step.get("step_num", 0)
                if step_num > total_steps:
                    total_steps = step_num

            return {
                "case_id": case_id,
                "trajectory_text": trajectory_text,
                "steps": steps,
                "total_steps": total_steps,
                "final_answer": final_answer,
                "model": config.deepseek.model_name,
                "temperature": config.deepseek.temperature,
            }
        except Exception as e:
            logger.log_error(f"Failed to generate MCQA trajectory: {str(e)}", case.get("case_id"))
            raise

    def _parse_trajectory(self, trajectory_text: str) -> List[Dict[str, str]]:
        import re

        steps = []
        lines = trajectory_text.split("\n")
        current_step = None
        current_content = []
        current_step_num = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Step "):
                match = re.match(r"Step (\d+):", line)
                if match:
                    current_step_num = int(match.group(1))
                continue

            thought_match = re.match(r"Thought\s*(\d+):\s*(.*)", line)
            action_match = re.match(r"Action\s*(\d+):\s*(.*)", line)
            observation_match = re.match(r"Observation\s*(\d+):\s*(.*)", line)
            final_match = re.match(r"Final Answer:\s*(.*)", line)

            if thought_match:
                if current_step:
                    steps.append({
                        "type": current_step,
                        "content": "\n".join(current_content).strip(),
                        "step_num": current_step_num,
                    })
                current_step = "thought"
                current_step_num = int(thought_match.group(1))
                current_content = [thought_match.group(2)]
            elif action_match:
                if current_step:
                    steps.append({
                        "type": current_step,
                        "content": "\n".join(current_content).strip(),
                        "step_num": current_step_num,
                    })
                current_step = "action"
                current_step_num = int(action_match.group(1))
                current_content = [action_match.group(2)]
            elif observation_match:
                if current_step:
                    steps.append({
                        "type": current_step,
                        "content": "\n".join(current_content).strip(),
                        "step_num": current_step_num,
                    })
                current_step = "observation"
                current_step_num = int(observation_match.group(1))
                current_content = [observation_match.group(2)]
            elif final_match:
                if current_step:
                    steps.append({
                        "type": current_step,
                        "content": "\n".join(current_content).strip(),
                        "step_num": current_step_num,
                    })
                current_step = "final_answer"
                current_content = [final_match.group(1)]
            else:
                if current_step:
                    current_content.append(line)

        if current_step:
            steps.append({
                "type": current_step,
                "content": "\n".join(current_content).strip(),
                "step_num": current_step_num,
            })

        return steps
