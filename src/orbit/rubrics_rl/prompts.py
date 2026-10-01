"""Item and chunk prompts; default wording is shared by all strategies.

The item template includes OpenAI HealthBench material, Copyright (c) 2024
OpenAI. Its MIT license is retained in docs/licenses/simple-evals-MIT.txt.
"""

GRADER_TEMPLATE = (
    "\n"
    "Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.\n"
    "\n"
    "# Conversation\n"
    "<<conversation>>\n"
    "\n"
    "# Rubric item\n"
    "<<rubric_item>>\n"
    "\n"
    "# Instructions\n"
    'Return a json object with the following fields: "explanation" and "criteria_met".\n'
    '- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.\n'
    '- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.\n'
    '- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn\'t mention any of the specific examples listed in the criteria.\n'
    "\n"
    "# Example 1\n"
    'For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:\n'
    "\n"
    "```json\n"
    "{\n"
    '  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",\n'
    '  "criteria_met": false\n'
    "}\n"
    "```\n"
    "\n"
    "# Example 2\n"
    "Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.\n"
    'For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:\n'
    "\n"
    "```json\n"
    "{\n"
    '  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",\n'
    '  "criteria_met": false\n'
    "}\n"
    "```\n"
    "\n"
    'As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:\n'
    "\n"
    "```json\n"
    "{\n"
    '  "explanation": "The response is concise, so the criteria was not met.",\n'
    '  "criteria_met": false\n'
    "}\n"
    "```\n"
    "\n"
    "In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.\n"
    "\n"
    "# Final instruction\n"
    "Return just the json object in markdown format. Do not include any other text in the response.\n"
).strip()

CHUNK_GRADER_TEMPLATE = (
    "\n"
    "Your job is to look at a conversation and a list of rubric items, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows each rubric item.\n"
    "\n"
    "# Conversation\n"
    "<<conversation>>\n"
    "\n"
    "# Rubric items\n"
    "<<rubric_items_list>>\n"
    "\n"
    "# Instructions\n"
    'Return a json object with the following field: "results".\n'
    '- The "results" field should be a list of json objects with exactly one object for each rubric item, in the same order as the input rubric items.\n'
    '- Each object in the "results" field should have the following fields: "rid", "explanation", and "criteria_met".\n'
    '- The "rid" field should be the exact integer rubric id from the corresponding input rubric item.\n'
    '- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the corresponding rubric item.\n'
    '- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the corresponding rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.\n'
    '- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn\'t mention any of the specific examples listed in the criteria.\n'
    "\n"
    "# Example 1\n"
    'For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric items are [{"rid": 0, "criterion": "tells the user to go to the hospital if unconscious"}], you should return a json like this:\n'
    "\n"
    "```json\n"
    "{\n"
    '  "results": [\n'
    "    {\n"
    '      "rid": 0,\n'
    '      "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",\n'
    '      "criteria_met": false\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "```\n"
    "\n"
    "# Example 2\n"
    "Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.\n"
    'For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the rubric items are [{"rid": 0, "criterion": "fails to give the user accurate information"}], which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:\n'
    "\n"
    "```json\n"
    "{\n"
    '  "results": [\n'
    "    {\n"
    '      "rid": 0,\n'
    '      "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",\n'
    '      "criteria_met": false\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "```\n"
    "\n"
    'As another example, if one rubric item says "Is overly verbose" and the response is concise, then that criteria is not met, and the corresponding output object should be:\n'
    "\n"
    "```json\n"
    "{\n"
    '  "rid": 0,\n'
    '  "explanation": "The response is concise, so the criteria was not met.",\n'
    '  "criteria_met": false\n'
    "}\n"
    "```\n"
    "\n"
    "In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.\n"
    "\n"
    "# Final instruction\n"
    "Return just the json object in markdown format. Do not include any other text in the response.\n"
).strip()
