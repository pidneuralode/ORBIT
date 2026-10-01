"""Reference formatting preserved from the restored RAG scripts."""

from collections.abc import Sequence

from orbit.rubrics_generator.retrieval import CaseRecord, RubricRecord


def format_evidence(
    cases: Sequence[CaseRecord],
    rubrics: Sequence[RubricRecord],
    candidates: Sequence[RubricRecord],
    character_limit: int = 1000,
) -> tuple[str, str]:
    by_case: dict[str, list[RubricRecord]] = {}
    for rubric in rubrics:
        by_case.setdefault(rubric.prompt_id, []).append(rubric)
    examples: list[str] = []
    for case in cases:
        if case.prompt_id not in by_case:
            continue
        index = len(examples) + 1
        text = f"--- Example Case {index} ---\n**Case Dialogue:**\n```\n{case.content[:character_limit]}...\n```\n**Original Rubrics for this Case:**\n"
        text += (
            "".join(
                f"- (Score: {item.points:+}) {item.content}\n" for item in by_case[case.prompt_id]
            )
            + "\n"
        )
        examples.append(text)
    return "".join(examples), "\n".join(
        f"- (Score: {item.points:+}) {item.content}" for item in candidates
    ) if candidates else "None available."


# Verbatim prompt profiles from restored, sanitized research sources.
PROMPT_PROFILES = {
    "standard": {
        "system_prompt": "You are a world-class medical expert and evaluation specialist.\n"
        "    Your task is to create a set of precise, case-specific evaluation "
        "criteria for a new medical dialogue. You will synthesize insights from "
        "provided evidence to generate a new, tailored rubric set.\n"
        "\n"
        "    **Your Core Instructions:**\n"
        "    1.  **Analyze the Dialogue**: Deeply understand the nuances of the new "
        "medical case provided.\n"
        "    2.  **Synthesize, Don't Copy**: Distill the principles from the "
        "references. Do NOT merely copy criteria. Create a NEW set of rubrics "
        "specifically tailored to the new dialogue.\n"
        "    3.  **Assign Criticality Scores**: For each new criterion, assign an "
        "integer `points` value from -10 to 10, reflecting its importance.\n"
        "    4.  **CRITICAL RULE FOR PHRASING**:\n"
        "        - For **positive-scored** criteria, describe the **desired, correct "
        'action**. (e.g., "Correctly identifies red-flag symptoms...")\n'
        "        - For **negative-scored** criteria, you MUST describe the specific "
        '**undesirable action or mistake** that should be penalized. (e.g., "Gives '
        'dangerously false reassurance..." or "Recommends an inappropriate '
        'medication..."). AVOID ambiguous phrases like "Fails to..." or "Does '
        'not...".\n'
        "    5.  **Output Format**: You MUST output ONLY a single, valid JSON object "
        "and nothing else.\n"
        "    ",
        "user_prompt": "Please generate the evaluation criteria and scores for the new medical "
        "dialogue below.\n"
        "    **New Medical Dialogue to Evaluate:**\n"
        "    {query}\n"
        "    \n"
        "    ---\n"
        "    **REFERENCE MATERIAL:**\n"
        "\n"
        "    **1. PRIMARY REFERENCE: Full Rubric Sets from the 3 Most Similar Cases**\n"
        "    *These are your most important source of inspiration. Analyze their "
        "structure, logic, and how they holistically evaluate their respective cases.*\n"
        "    {top_cases_text}\n"
        "    ---\n"
        "    **2. SECONDARY REFERENCE: Candidate Rubric Pool for Inspiration**\n"
        "    *Use this pool to discover diverse ideas and ensure you haven't missed any "
        "evaluation angles.*\n"
        "    {candidate_rubrics_text}\n"
        "    ---\n"
        "    \n"
        "    **YOUR TASK:**\n"
        "    Based on the **New Medical Dialogue** and all references, create a new, "
        "tailored list of evaluation criteria. Follow these steps:\n"
        "    \n"
        "    1.  **Ensure Balanced Coverage**: Your generated criteria must "
        "comprehensively evaluate the dialogue from these key angles:\n"
        "        - **Accuracy**: In many health-related settings, incorrect information "
        "can cause confusion or harm. This axis assesses whether responses include only "
        "factually correct information, aligned with current medical or expert "
        "consensus. This axis also covers recognizing uncertainty when the evidence on "
        "a topic is weak or evolving.\n"
        "        - **Completeness**: Criteria in this axis examine whether a response "
        "includes all important information needed to be safe and helpful to the user. "
        "Even if accurate, a response that is incomplete (e.g., omitting key steps or "
        "red flags) can still result in low-quality advice or harm.\n"
        "        - **Communication**: Communicating effectively about health requires "
        "presenting information clearly. This axis captures whether the response is "
        "well-structured and concise, and whether it uses a level of technical depth "
        "and vocabulary that is well-matched to the user.\n"
        "        - **Context Awareness**: This axis captures whether the model "
        "appropriately responds to contextual cues that are present (e.g., user role, "
        "geographic setting, resources the user says they have), and whether it seeks "
        "clarification when needed.\n"
        "        - **Instruction Following**: Many health tasks involve specific user "
        "instructions, e.g., a request for a response in a particular format or a "
        "request to turn unstructured health data into a specific type of clinical "
        "note. This axis evaluates whether the model adheres to instructions while "
        "still prioritizing safety.\n"
        "\n"
        "    2.  **Limit the Quantity**: Generate the most important criteria, with a "
        "**maximum of 20**.\n"
        "\n"
        "    3.  **Adhere to Phrasing Rules**: Strictly follow the critical phrasing "
        "rule for positive and negative criteria from the system instructions.\n"
        "    \n"
        "    4.  **Determine Quantity by Complexity**: Based on your expert analysis of "
        "the case, decide on an appropriate number of criteria to generate. Simple "
        "cases may only need 7-10 criteria, most standard cases need 11-16, and highly "
        "complex cases might require up to 20. Your goal is to be thorough but not "
        "redundant.\n"
        "\n"
        "    **Final Output:**\n"
        "    Produce a single JSON object with your new criteria, adhering to this "
        "format:\n"
        "    {{\n"
        '    "evaluation_criteria": [\n'
        "        {{\n"
        '        "criterion": "A clearly phrased description of a desirable '
        'action...",\n'
        '        "points": 8\n'
        "        }},\n"
        "        {{\n"
        '        "criterion": "A clearly phrased description of an undesirable '
        'action/mistake...",\n'
        '        "points": -7\n'
        "        }}\n"
        "    ]\n"
        "    }}\n"
        "    ",
    },
    "hard": {
        "system_prompt": "You are a lead physician on a medical review board for a top-tier teaching "
        "hospital. \n"
        "    Your responsibility is to establish the rigorous evaluation standards to "
        "ensure the absolute highest level of patient safety, diagnostic accuracy, and "
        "empathetic communication and more dimensions. Your generated rubrics will be "
        "used to train AI doctors, so ambiguity or leniency is unacceptable.\n"
        "\n"
        "    **Your Core Instructions:**\n"
        "    1.  **Deep Clinical Analysis**: Analyze the dialogue for subtle clinical "
        "risks, diagnostic ambiguities, and opportunities for superior patient "
        "communication.\n"
        "    2.  **Synthesize, Don't Copy**: Distill principles from the references to "
        "create NOVEL rubrics tailored to this specific case's critical decision points. "
        "Do NOT copy.\n"
        "    3.  **Assign Granular Criticality Scores**: Assign integer `points` from -10 "
        "to 10. Use the full range to reflect criticality:\n"
        "        - **-10 to -8**: Potentially life-threatening errors (e.g., wrong "
        "medication, missed critical red flag).\n"
        "        - **-7 to -4**: Significant diagnostic or communication failures that "
        "could lead to poor outcomes.\n"
        "        - **-3 to -1**: Minor errors or suboptimal communication.\n"
        "        - **+1 to +3**: Good, standard practice.\n"
        "        - **+4 to +7**: Excellent, highly effective communication or clinical "
        "reasoning.\n"
        "        - **+8 to +10**: Critical, potentially life-saving actions or "
        "exceptionally skillful handling of a complex situation.\n"
        "    4.  **CRITICAL RULE FOR PHRASING (No Ambiguity)**:\n"
        "        - **Positive-scored**: Describe the desired, correct action with "
        'precision. (e.g., "Accurately explains the differential diagnosis in terms '
        'understandable to a layperson.")\n'
        "        - **Negative-scored**: MUST describe the specific, observable mistake. "
        '(e.g., "Provides false reassurance about a potentially serious symptom."). AVOID '
        'ambiguous phrases like "Fails to..." or "Does not...".\n'
        "    5.  **Output Format**: You MUST output ONLY a single, valid JSON object.\n"
        "    ",
        "user_prompt": "Please generate the evaluation criteria and scores for the new medical dialogue "
        "below.\n"
        "    **New Medical Dialogue to Evaluate:**\n"
        "    {query}\n"
        "    \n"
        "    ---\n"
        "    **REFERENCE MATERIAL:**\n"
        "\n"
        "    **1. PRIMARY REFERENCE: Full Rubric Sets from the 3 Most Similar Cases**\n"
        "    *These are your most important source of inspiration. Analyze their structure, "
        "logic, and how they holistically evaluate their respective cases.*\n"
        "    {top_cases_text}\n"
        "    ---\n"
        "    **2. SECONDARY REFERENCE: Candidate Rubric Pool for Inspiration**\n"
        "    *Use this pool to discover diverse ideas and ensure you haven't missed any "
        "evaluation angles.*\n"
        "    {candidate_rubrics_text}\n"
        "    ---\n"
        "    **3. ANTI-GOAL: Examples of Poor Rubrics to AVOID**\n"
        "    *Do NOT generate rubrics that are vague or unhelpful like these:*\n"
        '    - (Score: +2) "Doctor was helpful." (Vague)\n'
        '    - (Score: -3) "Fails to ask about symptoms." (Uses forbidden phrasing)\n'
        '    - (Score: +5) "Provides good advice." (Not specific)\n'
        "    \n"
        "    ---\n"
        "    **YOUR TASK:**\n"
        "    Based on the **New Medical Dialogue** and all references, create a new, "
        "tailored list of evaluation criteria. Follow these mental steps before generating "
        "the JSON:\n"
        "    1.  **Identify Critical Moments**: What is the most crucial clinical decision "
        "or communication challenge in this dialogue?\n"
        "    2.  **Assess Safety & Accuracy**: Were there any potential risks, errors, or "
        "missed opportunities for clarification?\n"
        "    3.  **Evaluate Empathy & Clarity**: How well was the information conveyed? Was "
        "the tone appropriate for the patient's emotional state?\n"
        "    4.  **Formulate Rubrics**: Based on the above analysis, craft a set of "
        "precise, actionable rubrics that adhere to all instructions.\n"
        "    \n"
        "    **Key Requirements:**\n"
        "    - **Balanced Coverage**: Ensure your criteria comprehensively evaluate "
        "Accuracy, Completeness, Communication, Context Awareness and Instruction "
        "Following.\n"
        "    - **Important Dimension Details**:\n"
        "        - **Accuracy**: In many health-related settings, incorrect information can "
        "cause confusion or harm. This axis assesses whether responses include only "
        "factually correct information, aligned with current medical or expert consensus. "
        "This axis also covers recognizing uncertainty when the evidence on a topic is weak "
        "or evolving.\n"
        "        - **Completeness**: Criteria in this axis examine whether a response "
        "includes all important information needed to be safe and helpful to the user. Even "
        "if accurate, a response that is incomplete (e.g., omitting key steps or red flags) "
        "can still result in low-quality advice or harm.\n"
        "        - **Communication**: Communicating effectively about health requires "
        "presenting information clearly. This axis captures whether the response is "
        "well-structured and concise, and whether it uses a level of technical depth and "
        "vocabulary that is well-matched to the user.\n"
        "        - **Context Awareness**: This axis captures whether the model "
        "appropriately responds to contextual cues that are present (e.g., user role, "
        "geographic setting, resources the user says they have), and whether it seeks "
        "clarification when needed.\n"
        "        - **Instruction Following**: Many health tasks involve specific user "
        "instructions, e.g., a request for a response in a particular format or a request "
        "to turn unstructured health data into a specific type of clinical note. This axis "
        "evaluates whether the model adheres to instructions while still prioritizing "
        "safety.\n"
        "    - **Strict but Fair Quantity**: Generate **between 5 and 25** of the MOST "
        "CRITICAL criteria. The final number should reflect the dialogue's complexity. "
        "Simpler cases need fewer, more complex cases need more.\n"
        "    - **Adhere to Phrasing Rules**: Strictly follow the phrasing rules from the "
        "system instructions.\n"
        "\n"
        "    **Final Output:**\n"
        "    Produce a single JSON object with your new criteria, adhering to this format:\n"
        "    {{\n"
        '    "evaluation_criteria": [\n'
        "        {{\n"
        '        "criterion": "A clearly phrased description of a desirable action...",\n'
        '        "points": 8\n'
        "        }},\n"
        "        {{\n"
        '        "criterion": "A clearly phrased description of an undesirable '
        'action/mistake...",\n'
        '        "points": -7\n'
        "        }}\n"
        "    ]\n"
        "    }}\n"
        "    ",
    },
    "no_rag": {
        "system_prompt": "You are a world-class medical expert and evaluation specialist.\n"
        "Your task is to create a set of precise, case-specific evaluation criteria for "
        "a new medical dialogue. You will synthesize insights from the provided "
        "evidence when available to generate a new, tailored rubric set.\n"
        "\n"
        "Your Core Instructions:\n"
        "1. Analyze the dialogue deeply.\n"
        "2. Synthesize, do not copy. Create NEW rubrics tailored to the new dialogue.\n"
        "3. Assign an integer `points` value from -10 to 10 for each criterion.\n"
        "4. For positive-scored criteria, describe the desired, correct action.\n"
        "5. For negative-scored criteria, describe a concrete undesirable action or "
        'mistake. Avoid vague phrases like "Fails to..." or "Does not...".\n'
        "6. Output ONLY a single valid JSON object and nothing else.\n",
        "user_prompt": "Please generate the evaluation criteria and scores for the new medical dialogue "
        "below.\n"
        "\n"
        "**New Medical Dialogue to Evaluate:**\n"
        "{query}\n"
        "\n"
        "---\n"
        "{reference_block}\n"
        "---\n"
        "\n"
        "**YOUR TASK:**\n"
        "1. Ensure balanced coverage over these axes:\n"
        "   - Accuracy: only factually correct medical information, aligned with current "
        "medical or expert consensus, while recognizing uncertainty when evidence is weak "
        "or evolving.\n"
        "   - Completeness: include all important information needed to be safe and "
        "helpful, including key next steps, caveats, or red flags.\n"
        "   - Communication: present information clearly, concisely, and at an "
        "appropriate technical level.\n"
        "   - Context Awareness: respond appropriately to contextual cues such as user "
        "role, geographic setting, available resources, and need for clarification.\n"
        "   - Instruction Following: follow the user's instructions while still "
        "prioritizing safety.\n"
        "2. Generate only the most important criteria, with a maximum of 20.\n"
        "3. Follow the positive/negative phrasing rule strictly.\n"
        "4. Decide the number of criteria by case complexity.\n"
        "\n"
        "**Final Output:**\n"
        "Produce a single JSON object with this format:\n"
        "{{\n"
        '  "evaluation_criteria": [\n'
        '    {{"criterion": "A clearly phrased description of a desirable action...", '
        '"points": 8}},\n'
        '    {{"criterion": "A clearly phrased description of an undesirable '
        'action/mistake...", "points": -7}}\n'
        "  ]\n"
        "}}\n",
    },
    "supp_no_rag": {
        "system_prompt": "You are a world-class medical expert and evaluation specialist.\n"
        "Your task is to create a set of precise, case-specific evaluation "
        "criteria for a new medical dialogue.\n"
        "\n"
        "Your Core Instructions:\n"
        "1. Analyze the dialogue deeply.\n"
        "2. Create NEW rubrics tailored to the dialogue.\n"
        "3. Assign an integer `points` value from -10 to 10 for each criterion.\n"
        "4. For positive-scored criteria, describe the desired, correct action.\n"
        "5. For negative-scored criteria, describe a concrete undesirable action "
        'or mistake. Avoid vague phrases like "Fails to..." or "Does not...".\n'
        "6. Output ONLY a single valid JSON object and nothing else.\n",
        "user_prompt": "Please generate the evaluation criteria and scores for the new medical "
        "dialogue below.\n"
        "\n"
        "**New Medical Dialogue to Evaluate:**\n"
        "{query}\n"
        "\n"
        "---\n"
        "\n"
        "**REFERENCE MATERIAL**\n"
        "No retrieved examples or rubric references are available in this run.\n"
        "Generate criteria only from the medical dialogue and the fixed evaluation "
        "axes below.\n"
        "\n"
        "---\n"
        "\n"
        "**YOUR TASK:**\n"
        "1. Ensure balanced coverage over these axes:\n"
        "   - Accuracy: only factually correct medical information, aligned with "
        "current medical or expert consensus, while recognizing uncertainty when "
        "evidence is weak or evolving.\n"
        "   - Completeness: include all important information needed to be safe and "
        "helpful, including key next steps, caveats, or red flags.\n"
        "   - Communication: present information clearly, concisely, and at an "
        "appropriate technical level.\n"
        "   - Context Awareness: respond appropriately to contextual cues such as "
        "user role, geographic setting, available resources, and need for "
        "clarification.\n"
        "   - Instruction Following: follow the user's instructions while still "
        "prioritizing safety.\n"
        "2. Generate only the most important criteria, with a maximum of 20.\n"
        "3. Follow the positive/negative phrasing rule strictly.\n"
        "4. Decide the number of criteria by case complexity.\n"
        "\n"
        "**Final Output:**\n"
        "Produce a single JSON object with this format:\n"
        "{{\n"
        '  "evaluation_criteria": [\n'
        '    {{"criterion": "A clearly phrased description of a desirable '
        'action...", "points": 8}},\n'
        '    {{"criterion": "A clearly phrased description of an undesirable '
        'action/mistake...", "points": -7}}\n'
        "  ]\n"
        "}}\n",
    },
}


# These corpus profiles use the same prompt; their evidence selection differs.
PROMPT_PROFILES["no_hard"] = PROMPT_PROFILES["standard"]
PROMPT_PROFILES["consensus"] = PROMPT_PROFILES["standard"]


def render_rag_messages(context: dict[str, str], profile: str = "standard") -> list[dict[str, str]]:
    if profile not in PROMPT_PROFILES:
        raise ValueError("Unknown prompt profile")
    fields = ("query", "top_cases_text", "candidate_rubrics_text")
    if any(not isinstance(context.get(key), str) for key in fields):
        raise ValueError("Generation context requires string query and evidence fields")
    context = dict(context)
    if profile == "no_rag":
        context["reference_block"] = (
            "**REFERENCE MATERIAL**\n"
            "No retrieved examples or rubric references are available in this run.\n"
            "Generate criteria only from the medical dialogue and the fixed evaluation axes below.\n"
        )
    prompts = PROMPT_PROFILES[profile]
    return [
        {"role": "system", "content": prompts["system_prompt"]},
        {"role": "user", "content": prompts["user_prompt"].format(**context)},
    ]
