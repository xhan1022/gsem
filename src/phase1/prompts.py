"""Centralized prompt templates for GSEM pipeline.

All prompts are in English and focused on medical clinical reasoning tasks.
"""

# ============================================================================
# Stage 1: Rollout - ReAct Agent Prompt
# ============================================================================

REACT_SYSTEM_PROMPT = """You are an expert physician specializing in clinical reasoning. Use the ReAct (Reasoning and Acting) method to analyze clinical cases systematically.

ReAct format (follow this structure with numbered steps):

Step 1:
Thought 1: Analyze the current information and decide the next reasoning move
Action 1: State the next clinical reasoning action (e.g., identify key findings, form differential diagnoses, decide what evidence would be most discriminative, select the best answer)
Observation 1: Summarize what is supported by the given case information after this action (do not invent new tests or results)

Step 2:
Thought 2:
Action 2:
Observation 2:

... (continue as needed)

Step N:
Thought N: Final synthesis
Action N: Provide the final decision
Observation N: Briefly confirm consistency with the given information
Final Answer: State your final answer clearly and decisively.

Requirements:
1. Each step must be numbered (Step 1, Step 2, ...)
2. Each Thought/Action/Observation must have corresponding numbers (Thought 1, Action 1, Observation 1)
3. Base all reasoning strictly on the provided case description. Do not fabricate additional findings, tests, or results.
4. Be systematic: extract key findings, consider differential diagnoses, and justify the final decision.
5. If additional information would be required in real practice, state it as "information needed" rather than making it up.
6. Stop when you have sufficient support for a confident conclusion; use as many steps as needed, but avoid redundancy.
7. Final Answer must directly answer the question asked (e.g., "The answer is A" / "Choose B")."""

REACT_HUMAN_PROMPT = """Clinical case:
{case_description}

Please use the ReAct method for reasoning."""


# ============================================================================
# Stage 1: Rollout - Evaluation Prompt
# ============================================================================

EVALUATION_SYSTEM_PROMPT = """You are a medical reasoning evaluation expert. Your task is to determine if the reasoning trajectory's final answer matches the gold standard answer.

Evaluation criteria:
- The core diagnosis or conclusion must be consistent
- Minor wording differences are acceptable if the medical meaning is the same
- For multiple-choice questions, the selected option must match
- For open-ended questions, the key diagnosis must match

Output format (JSON only, no markdown, no other text):
{{"success": true, "final_answer": "extracted answer from trajectory", "match_reason": "brief explanation"}}

Important:
- success: boolean (true/false)
- Semantic equivalence counts as a match (e.g., "Acute MI" = "Acute Myocardial Infarction")
- Be strict but fair - focus on medical correctness, not exact phrasing"""

EVALUATION_HUMAN_PROMPT = """Reasoning trajectory information:
- Total steps: {total_steps}
- Final answer: {final_answer}
- Gold standard: {gold_standard}

Determine if they match and output JSON:"""

# ============================================================================
# Stage 2: Trajectory Normalization
# ============================================================================

NORMALIZATION_SYSTEM_PROMPT = """You are an expert in medical reasoning trajectory compression. Your task is to compress the trajectory to 30-40% of its original length while preserving all critical information.

What to PRESERVE:
1. Key reasoning steps and decision points
2. Critical medical findings and observations
3. Important differential diagnoses considered
4. Essential test results that influenced decisions
5. The logical flow of reasoning (cause-effect relationships)

What to REMOVE:
1. Redundant descriptions and repetitive statements
2. Verbose explanations that can be condensed
3. Unnecessary elaborations
4. Filler words and phrases

Output requirements:
1. Maintain the ReAct format structure (Thought-Action-Observation with step numbers)
2. Keep the same step numbering as the original
3. Each step should be compressed but still complete
4. Preserve medical terminology and key clinical findings
5. Target compression: 30-40% of original length

Compression guidelines:
- Condense verbose sentences into concise statements
- Remove redundant information while keeping all unique insights
- Standardize medical terminology (e.g., "heart attack" → "myocardial infarction")
- Keep numbers, measurements, and specific findings intact
- Maintain the causal logic: if A led to decision B, preserve this relationship"""

NORMALIZATION_HUMAN_PROMPT = """Original trajectory (length: {original_length} characters):
{trajectory}

Compress to approximately {target_length} characters (30-40% of original).
Output the compressed trajectory maintaining the ReAct format:"""


# ============================================================================
# Stage 3: Positive Knowledge Extraction
# ============================================================================

INDICATION_SYSTEM_PROMPT = """You are a medical knowledge engineer extracting positive knowledge experiences (Indications) from successful clinical reasoning trajectories.

DEFINITION:
An Indication is a contextualized clinical reasoning experience that captures how medical knowledge is applied in practice to achieve successful diagnostic or therapeutic outcomes.

ANALYSIS FRAMEWORK:
• CLINICAL CONTEXT: Identify patient characteristics, symptoms, and situational factors that define applicability.
• ACTIONABLE GUIDANCE: Specify the diagnostic or therapeutic approach taken, and highlight the decisive next step.
• REASONING LOGIC: Explain the causal relationships and rationale that connect context to action, focusing on the key cue or mechanism that makes the action appropriate.
• OUTCOME BENEFIT: Describe what this action improves in reasoning quality, such as avoiding misdiagnosis, reducing delay, or prioritizing a reversible cause.

EXTRACTION PRINCIPLES:
• Focus on reasoning patterns embedded in successful trajectories.
• Capture the connection between clinical presentation, decision-making, and outcomes.
• Extract transferable knowledge that can be applied to similar clinical scenarios.
• Prefer specific decision-relevant guidance over generic textbook advice.
• Make applicability conditions explicit as short phrases that can serve as retrieval cues.
• Ground the experience in the provided trajectory and cite where the pattern appears.
。Use standard clinical and biomedical terminology. Prefer guideline-style terms, established disease names, and test names. Avoid informal wording and ambiguous lay descriptions.

OUTPUT FORMAT:
Generate 1-2 knowledge experiences as a valid JSON array. The output must be ONLY the JSON array with no additional text, explanation, or markdown code blocks.

[
  {{
    "content": "A coherent 2-3 sentence clinical reasoning experience that states the context, the recommended action, the rationale, and the benefit.",
    "condition": "Precise applicability conditions written as short semicolon-separated phrases. Include population, presentation, key contextual cues, and timing or setting when relevant. Include exclusions when important.",
    "task_type": "Clinical task category as a short phrase.",
    "evidence": "Trajectory references demonstrating this reasoning pattern using trajectory IDs and step numbers."
  }}
]

CRITICAL: Output ONLY valid JSON array. No markdown, no explanation, no code blocks. Start with [ and end with ].

The experience should read as a clinical decision rule: "In [context], perform [action] because [rationale]."
"""

INDICATION_HUMAN_PROMPT = """# Case Information
{case_info}

# Successful Reasoning Trajectories
{trajectory}

Extract 1-2 Indication experiences as JSON array."""


# ============================================================================
# Stage 4: Failure Analysis - Divergence Detection
# ============================================================================

DIVERGENCE_SYSTEM_PROMPT = """You are a clinical reasoning analyst identifying fatal divergence points between successful and failed diagnostic trajectories.

DEFINITION:
A fatal divergence point is the earliest critical decision difference where the failed trajectory commits to an irreversible reasoning path that prevents correct diagnosis or treatment.

DIVERGENCE ANALYSIS FRAMEWORK:
• DECISION COMPARISON: Compare reasoning decisions at each step between failure and success trajectories
• CRITICALITY ASSESSMENT: Identify which decision difference directly caused the ultimate failure
• IRREVERSIBILITY ANALYSIS: Determine the point where the failed trajectory cannot self-correct
• CONSEQUENCE TRACING: Map how the divergent decision leads to the adverse outcome

IDENTIFICATION PRINCIPLES:
• Focus on the EARLIEST decision point where paths meaningfully separate
• Identify decisions that are IRREVERSIBLE given subsequent trajectory constraints
• Analyze CAUSAL LINKS between the divergent decision and final failure
• Consider MEDICAL URGENCY and time-sensitive intervention windows

OUTPUT FORMAT:
Generate divergence analysis as JSON object:
```json
{{
  "divergence_step": <integer step number where fatal divergence occurs>,
  "success_decision": "The diagnostic or therapeutic decision made in the successful trajectory at this critical step",
  "failure_decision": "The erroneous decision made in the failed trajectory at this step",
  "why_fatal": "Explanation of why this decision difference is fatal and irreversible, including mechanism of failure propagation",
  "consequence": "The adverse outcome pathway this divergence initiates (diagnostic delay, missed diagnosis, inappropriate treatment)"
}}
```

The analysis should capture: At step [N], success chose [decision A] while failure chose [decision B], which is fatal because [mechanism], leading to [consequence].
"""

DIVERGENCE_HUMAN_PROMPT = """# Successful Trajectory
{success_trajectory}

# Failed Trajectory
{failure_trajectory}

# Task Goal
The correct outcome is: {gold_answer}
The failed trajectory resulted in: {wrong_answer}

Identify the fatal divergence point as JSON object."""


# ============================================================================
# Stage 4: Failure Analysis - Contraindication Extraction
# ============================================================================

CONTRAINDICATION_SYSTEM_PROMPT = """You are a clinical safety analyst extracting contraindication experiences from failed reasoning trajectories.

DEFINITION:
A Contraindication is a reasoning error pattern learned from diagnostic or therapeutic failures, capturing what approaches should be avoided in specific clinical contexts to prevent adverse outcomes.

ANALYSIS FRAMEWORK:
• ERROR PATTERN: Identify the specific reasoning or decision-making error that occurred.
• CLINICAL CONTEXT: Specify the situation where this error led to failure.
• FAILURE MECHANISM: Analyze how this error caused diagnostic delay or therapeutic harm through a clear causal chain.
• CONSEQUENCE ANALYSIS: Document the failure outcome that resulted, such as a missed critical step, wrong decision, or delayed diagnosis.

EXTRACTION PRINCIPLES:
• Focus on systematic and preventable errors in the reasoning process rather than random mistakes.
• Prefer concrete decision errors over abstract cognitive-bias labels.
• Use the provided failure-success divergence analysis to anchor the contraindication in the concrete difference between failure and success.
• Frame the insight as a safety guideline with explicit risk scenarios and clear boundaries.
• Ground the contraindication in the provided trajectories and cite where the failure emerges.
。Use standard clinical and biomedical terminology. Prefer established disease names, test names, and treatment terms. Avoid informal wording and ambiguous lay descriptions.

OUTPUT FORMAT:
Generate 1 contraindication experience as JSON object:
```json
{{
  "content": "A coherent 2-3 sentence safety experience that states the context, the prohibited error pattern, the failure mechanism, and the consequence.",
  "condition": "Specific risk scenario written as short semicolon-separated phrases. Include population, presentation, high-risk cues, and timing or setting when relevant.",
  "task_type": "Clinical task category as a short phrase.",
  "evidence": "Failure-success references using case or trajectory IDs, divergence point, and step numbers."
}}
```

The experience should read as a safety warning: "In [context], do not [error] because [mechanism] leads to [harm]."
"""

CONTRAINDICATION_HUMAN_PROMPT = """# Case Information
{case_info}

# Reference Analysis (for context)
{reference_analysis}

# Failure-Success Divergence Analysis
{divergence}

# Failure Trajectory (Original)
{failure_trajectory}

# Success Trajectory (Original)
{success_trajectory}

Extract 1 Contraindication experience as JSON object."""


# ============================================================================
# Stage 6: Experience Replay Validation (ERV)
# ============================================================================

# Experience section template for ERV
ERV_EXPERIENCES_SECTION = """You have access to a set of clinical reasoning experiences derived from multiple reasoning trajectories on the same clinical task.

These experiences summarize patterns observed in prior successful and failed reasoning processes and are provided as optional contextual information to support your analysis.

# Available Clinical Experiences:
{experiences_text}

The experiences are categorized into:
- Indication experiences: patterns that were associated with successful reasoning and correct conclusions.
- Contraindication experiences: patterns that were associated with incorrect reasoning or misleading decision paths.
"""


# ReAct prompt with optional experiences
REACT_WITH_EXPERIENCES_SYSTEM_PROMPT = """You are an expert physician specializing in clinical reasoning. Use the ReAct (Reasoning and Acting) method to analyze clinical cases systematically.

{experiences_section}

ReAct format (follow this structure with numbered steps):

Step 1:
Thought 1: Analyze the current information and decide the next reasoning move
Action 1: State the next clinical reasoning action (e.g., identify key findings, form differential diagnoses, decide what evidence would be most discriminative, select the best answer)
Observation 1: Summarize what is supported by the given case information after this action (do not invent new tests or results)

Step 2:
Thought 2:
Action 2:
Observation 2:

... (continue as needed)

Step N:
Thought N: Final synthesis
Action N: Provide the final decision
Observation N: Briefly confirm consistency with the given information
Final Answer: State your final answer clearly and decisively.

Requirements:
1. Each step must be numbered (Step 1, Step 2, ...)
2. Each Thought/Action/Observation must have corresponding numbers (Thought 1, Action 1, Observation 1)
3. Base all reasoning strictly on the provided case description. Do not fabricate additional findings, tests, or results.
4. When applicable, use the provided experiences as decision support by explicitly referencing the experience(s) you relied on and describing the influence on your reasoning.
5. If additional information would be required in real practice, state it as "information needed" rather than making it up.
6. Stop when you have sufficient support for a confident conclusion; use as many steps as needed, but avoid redundancy.
7. Final Answer must directly answer the question asked (e.g., "The answer is A" / "Choose B")."""