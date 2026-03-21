"""Prompts for GSEM Phase 2: Experience Graph Construction."""

# ============================================================================
# Entity Extraction Prompt - Core Entities with Roles
# ============================================================================

CORE_ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are extracting a compact set of medical decision-structure entities from a reusable clinical reasoning experience.

Goal:
Extract only the most decision-driving medical entities that define the clinical situation and the actionable strategy, while keeping the entity list small and easy to normalize.

Role schema (fixed):
- Condition: diagnoses, symptoms, findings, clinical states, patient status.
- Constraint: contraindications, feasibility limits, special risks, "cannot / not suitable" factors.
- Action: diagnostic tests, treatments, interventions, procedures, management steps.
- Rationale: medical reasoning basis, mechanism, justification, key rationale statements.
- Outcome: intended clinical goal or effect (e.g., prevent complication, control symptoms, enable recovery).

Role schema (fixed) and how to choose:
- Condition: keep 1–2 anchors that determine whether the strategy applies (typically one core diagnosis, key presentation, or key population feature).
- Constraint: keep at most 1 anchor that makes the default approach infeasible or substantially increases risk; if none, omit this role.
- Action: keep 1–2 anchors that truly change the decision (the main action plus one key alternative or critical adjunct if needed).
- Rationale: keep at most 1 anchor that explains why the strategy holds (mechanism/causal link/critical disambiguation); if none, omit this role.
- Outcome: keep at most 1 anchor describing the intent or expected benefit (e.g., control output, avoid misdiagnosis, enable rehabilitation); if none, omit this role.

Extraction rules:
- Extract only medically meaningful, decision-driving anchor entities; do not aim for exhaustive coverage.
- Use standard medical terminology; avoid generic words.
- **CRITICAL: Each entity MUST be a noun or noun phrase of EXACTLY 1-3 words. No exceptions.**
  - Good examples: "myocardial infarction" (2 words), "heart failure" (2 words), "COPD" (1 word)
  - Bad examples: "optimize wound healing environment" (4 words, too long), "prevent aspiration" (verb phrase, not noun)
- Do not duplicate the same meaning across roles.
- Each entity must be assigned to exactly one role from the fixed schema.
- Keep the total set compact (typically 5–8 entities); omit roles not supported by the experience rather than forcing them.

Quick role boundary reminders:
- Condition vs Constraint: Condition describes the clinical situation; Constraint describes what blocks or makes the default plan unsafe/infeasible under that situation.
- Rationale vs Outcome: Rationale is the medical justification for the action; Outcome is the intended goal/benefit after taking the action.

Output format:
Return a JSON object with a single field "core_entities" as a list.
Each item must have "entity" and "role".

Example format:
{{
  "core_entities": [
    {{"entity": "...", "role": "Condition"}},
    {{"entity": "...", "role": "Action"}}
  ]
}}"""

CORE_ENTITY_EXTRACTION_HUMAN_PROMPT = """## Condition
{condition}

## Content
{content}

Extract the core decision-structure entities and roles as specified in the system prompt.

Return a valid JSON object with field 'core_entities' containing a list of entity-role pairs.
Output only the JSON, no other text."""


# ============================================================================
# Structure Extraction Prompt - Role-Edges
# ============================================================================

ROLE_EDGE_EXTRACTION_SYSTEM_PROMPT = """You are extracting the decision-flow structure (reasoning skeleton) of a clinical experience using role-to-role edges.

Your task:
Based on the provided core entities (with roles) and the experience text, identify the key decision logic pattern of the experience.

Role schema (fixed):
- Condition
- Constraint
- Action
- Rationale
- Outcome

A role-edge represents a meaningful transition in the decision process.

Allowed role-edges:

Core skeleton edges:

• Condition→Action
  Clinical context directly triggers a strategy.

• Condition→Condition
  One disease or clinical state leads to a complication or secondary condition.

• Condition→Constraint
  The condition introduces a limitation or infeasibility.

• Constraint→Action
  A limitation modifies or redirects the strategy.

• Constraint→Rationale
  A constraint itself constitutes a medical reasoning basis.

• Constraint→Outcome
  A limitation directly determines the target goal direction.

• Action→Outcome
  The action aims to achieve a specific clinical goal.

• Action→Rationale
  The action is justified by medical reasoning.

• Action→Constraint
  An intervention introduces a new limitation or risk.

• Condition→Outcome
  The condition directly implies a target goal.

• Condition→Rationale  
  The clinical context motivates a key rationale before action.

• Rationale→Action  
  The rationale directly determines the action.

• Rationale→Outcome
  The rationale explains the expected benefit.

• Rationale→Constraint
  A mechanism or reasoning basis reveals a limitation or risk.

• Action→Action
  The strategy is sequential (multi-step management).

Extraction rules:

- Only use edges from the allowed list above.
- Do NOT connect roles mechanically.
- Include an edge only if it reflects a clear reasoning transition in the text.
- Prefer a minimal but expressive skeleton (typically 2–6 edges).
- The structure should capture the reasoning template, not every possible pair.

Output format:

Return a JSON object with TWO fields:

{{
  "role_edges": [
    "Condition→Constraint",
    "Constraint→Action",
    "Action→Outcome"
  ],
  "entity_edges": [
    {{
      "edge": "Condition→Constraint",
      "from_entity": "<entity from core_entities>",
      "to_entity": "<entity from core_entities>"
    }}
  ]
}}

Requirements:

- Every role_edge MUST be grounded by at least one entity_edge of the same type. Do NOT output a role_edge unless you can provide at least one concrete entity pair to support it.
- Every entity_edge must correspond to one of the role_edges.
- "from_entity" and "to_entity" must be copied exactly from the provided core entities.
- Do NOT invent new entities.
- Multiple entity pairs may map to the same role-edge if necessary.
- Keep the structure compact and clean.

Output only valid JSON. No explanations.
"""


ROLE_EDGE_EXTRACTION_HUMAN_PROMPT = """## Core Entities with Roles
{core_entities_json}

## Condition
{condition}

## Content
{content}

Extract the role-edge decision-flow structure and ground each role-edge to specific entity pairs from the core entities.

Return a valid JSON object with:
- "role_edges"
- "entity_edges"

Output only the JSON, no other text.
"""


# ============================================================================
# Semantic Similarity Prompt
# ============================================================================

SEMANTIC_SIMILARITY_SYSTEM_PROMPT = """You are evaluating the semantic similarity between two clinical reasoning experiences.

Your task:
Given two experiences (A and B), determine how similar they are in terms of medical decision-making patterns, clinical strategies, and reasoning logic.

Evaluation criteria:
- Clinical context similarity: Do they address similar medical situations, conditions, or patient populations?
- Strategy similarity: Do they use similar diagnostic or treatment approaches?
- Reasoning logic: Do they follow similar decision-making patterns or clinical reasoning?
- Transferability: Would insights from one experience be applicable to the other?

Scoring scale:
- 0.0: Completely unrelated (different domains, conditions, or strategies)
- 0.3: Weakly related (share some general medical principles but different contexts)
- 0.5: Moderately related (share some clinical aspects or reasoning patterns)
- 0.7: Highly related (similar conditions, strategies, or decision patterns)
- 1.0: Nearly identical (same condition and strategy with minor variations)

Focus on semantic and conceptual similarity, not just keyword overlap.

Output format:
Return a JSON object with two fields:
{{
  "similarity": <float between 0.0 and 1.0>,
  "reason": "<brief explanation in 1-2 sentences>"
}}"""

SEMANTIC_SIMILARITY_HUMAN_PROMPT = """## Experience A
Condition: {condition_a}
Content: {content_a}

## Experience B
Condition: {condition_b}
Content: {content_b}

Evaluate the semantic similarity between these two experiences.

Return a valid JSON object with 'similarity' (float) and 'reason' (string).
Output only the JSON, no other text."""
