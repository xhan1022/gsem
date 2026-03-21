"""ReAct Agent for trajectory generation using DeepSeek."""
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.shared.config import config
from src.shared.logger import logger
from ..prompt_provider import REACT_SYSTEM_PROMPT, REACT_HUMAN_PROMPT


class ReActAgent:
    """ReAct Agent for clinical reasoning."""

    def __init__(self):
        """Initialize ReAct Agent with DeepSeek."""
        self.llm = ChatOpenAI(
            model=config.deepseek.model_name,
            temperature=config.deepseek.temperature,
            openai_api_key=config.deepseek.api_key,
            openai_api_base=config.deepseek.base_url,
        )

    def generate_trajectory(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single trajectory for a case.

        Args:
            case: Clinical case data

        Returns:
            Trajectory with reasoning steps
        """
        try:
            # Extract case information
            case_description = case.get("task", case.get("description", ""))
            case_id = case.get("id", case.get("case_id", "unknown"))
            task_type = case.get("task_type", "diagnosis")

            # Construct task instruction based on task_type
            task_instructions = {
                "diagnosis": "Based on the following clinical case, provide the most likely diagnosis.",
                "treatment": "Based on the following clinical case, provide the most appropriate treatment plan."
            }
            task_instruction = task_instructions.get(task_type, "Analyze the following clinical case.")

            # Construct full prompt with task instruction
            full_prompt = f"{task_instruction}\n\n{case_description}\n\nPlease use the ReAct method for reasoning."

            messages = [
                SystemMessage(content=REACT_SYSTEM_PROMPT),
                HumanMessage(content=full_prompt)
            ]

            # Generate response
            response = self.llm.invoke(messages)
            trajectory_text = response.content

            # Parse trajectory into steps
            steps = self._parse_trajectory(trajectory_text)

            # Extract final answer and total steps
            final_answer = ""
            total_steps = 0
            for step in steps:
                if step["type"] == "final_answer":
                    final_answer = step["content"]
                step_num = step.get("step_num", 0)
                if step_num > total_steps:
                    total_steps = step_num

            trajectory = {
                "case_id": case_id,
                "trajectory_text": trajectory_text,
                "steps": steps,
                "total_steps": total_steps,
                "final_answer": final_answer,
                "model": config.deepseek.model_name,
                "temperature": config.deepseek.temperature
            }

            return trajectory

        except Exception as e:
            logger.log_error(f"Failed to generate trajectory: {str(e)}", case.get("case_id"))
            raise

    def _parse_trajectory(self, trajectory_text: str) -> List[Dict[str, str]]:
        """Parse trajectory text into structured steps.

        Args:
            trajectory_text: Raw trajectory text

        Returns:
            List of steps with type, content, and step number
        """
        steps = []
        lines = trajectory_text.split('\n')
        current_step = None
        current_content = []
        current_step_num = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for step markers with numbers
            if line.startswith("Step "):
                # Extract step number
                import re
                match = re.match(r'Step (\d+):', line)
                if match:
                    current_step_num = int(match.group(1))
                continue

            # Check for thought/action/observation with numbers
            thought_match = re.match(r'Thought\s*(\d+):\s*(.*)', line)
            action_match = re.match(r'Action\s*(\d+):\s*(.*)', line)
            observation_match = re.match(r'Observation\s*(\d+):\s*(.*)', line)
            final_match = re.match(r'Final Answer:\s*(.*)', line)

            if thought_match:
                if current_step:
                    steps.append({
                        "type": current_step,
                        "content": "\n".join(current_content).strip(),
                        "step_num": current_step_num
                    })
                current_step = "thought"
                current_step_num = int(thought_match.group(1))
                current_content = [thought_match.group(2)]
            elif action_match:
                if current_step:
                    steps.append({
                        "type": current_step,
                        "content": "\n".join(current_content).strip(),
                        "step_num": current_step_num
                    })
                current_step = "action"
                current_step_num = int(action_match.group(1))
                current_content = [action_match.group(2)]
            elif observation_match:
                if current_step:
                    steps.append({
                        "type": current_step,
                        "content": "\n".join(current_content).strip(),
                        "step_num": current_step_num
                    })
                current_step = "observation"
                current_step_num = int(observation_match.group(1))
                current_content = [observation_match.group(2)]
            elif final_match:
                if current_step:
                    steps.append({
                        "type": current_step,
                        "content": "\n".join(current_content).strip(),
                        "step_num": current_step_num
                    })
                current_step = "final_answer"
                current_content = [final_match.group(1)]
            else:
                if current_step:
                    current_content.append(line)

        # Add last step
        if current_step:
            steps.append({
                "type": current_step,
                "content": "\n".join(current_content).strip(),
                "step_num": current_step_num
            })

        return steps
