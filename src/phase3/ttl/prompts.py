"""TTL prompts — online (single-trajectory) experience extraction."""

# ============================================================================
# Step 3: Online Contraindication Extraction (single failed trajectory)
# ============================================================================

ONLINE_CONTRAINDICATION_SYSTEM_PROMPT = """You are a clinical safety analyst extracting a contraindication experience from a single failed reasoning trajectory.

DEFINITION:
A Contraindication is a reasoning error pattern learned from a diagnostic or therapeutic failure, capturing what approach should be avoided in the specific clinical context to prevent the wrong outcome.

EXTRACTION PRINCIPLES:
• Identify the concrete reasoning or decision error that led to the wrong answer.
• Frame as a safety warning grounded in this specific failure.
• Use standard clinical and biomedical terminology.
• Keep the experience concise (2-3 sentences) and actionable.

OUTPUT FORMAT:
Return a JSON object only — no markdown, no extra text:
{{
  "content": "In [context], do not [error] because [mechanism] leads to [wrong outcome].",
  "condition": "Specific risk scenario as short semicolon-separated phrases.",
  "task_type": "Clinical task category as a short phrase.",
  "evidence": "Reference to the failed trajectory (e.g., 'failed inference on case X, wrong answer Y vs gold Z')."
}}"""

ONLINE_CONTRAINDICATION_HUMAN_PROMPT = """# Case Information
{case_info}

# Failed Trajectory
{failure_trajectory}

# Wrong Answer (agent output)
{wrong_answer}

# Gold Standard Answer
{gold_answer}

Extract 1 Contraindication experience as JSON object."""
