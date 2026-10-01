"""Legacy ordered-array experiment prompt; this variant returns raw points."""

BATCH_GRADER_TEMPLATE = """Your job is to look at a conversation and a LIST of rubric items, and for EACH item, score the last turn (i.e., the last assistant response) on how well it follows that rubric item.

# Conversation
<<conversation>>

# Rubric Items (as a JSON array of strings)
<<rubric_items_list>>

# Instructions
Return a SINGLE JSON ARRAY where each object corresponds to a rubric item from the input list.
- The returned JSON array MUST contain exactly one object for each rubric item in the input list, IN THE SAME ORDER.
- Each object in the array must have the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the corresponding rubric item.
- The "criteria_met" field should be a boolean (true or false).

---
# Logic for Evaluating a Single Rubric Item (Apply this logic to each item in the list)

- If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true if all of the criteria are met.
- Exception: If a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria.

# Example 1 (Logic)
If the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", the corresponding object in the output array should be:
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}

# Example 2 (Logic for negative criteria)
For criteria with negative points (e.g., "fails to give the user accurate information"), a good response should be classified as `false` because it does not meet the undesirable criteria.
If the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", the corresponding object should be:
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
---

# Batch Input and Output Format Example

## Example Conversation:
"user: I have a headache and a fever. assistant: You should take some rest and drink water. If it gets worse, see a doctor."

## Example Rubric Items List (Input):
[
    "Recommends seeing a doctor.",
    "Gives dangerous advice.",
    "Provides at least two self-care tips."
]

## Expected JSON Array Output:
[
    {
        "explanation": "The assistant advised seeing a doctor if the symptoms get worse, which meets the criteria.",
        "criteria_met": true
    },
    {
        "explanation": "The advice to rest and drink water is safe and standard, so it does not meet the criteria of giving dangerous advice.",
        "criteria_met": false
    },
    {
        "explanation": "The assistant provided two self-care tips: 'take some rest' and 'drink water'.",
        "criteria_met": true
    }
]

# Final instruction
Return just the single JSON array in a markdown format. Do not include any other text in the response."""
