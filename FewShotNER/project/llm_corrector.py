from groq import Groq

# Configure Groq API
API_KEY = "YOUR_KEY"
client = Groq(api_key=API_KEY)

def correct_label(sentence, entity, metric_label,bert_label,bert_confidence, fewshot_examples):
    """
    Send suspicious prediction to LLM
    and get corrected entity label.
    """

    # Build Few-Shot Context
    fewshot_context = ""
    for idx, example in enumerate(fewshot_examples, start=1):
        fewshot_context += f"""
Example {idx}:
Sentence: "{example['sentence']}"
Entity: "{example['entity']}"
Correct Label: "{example['label']}"

"""

    # Structured Few-Shot Prompt
    prompt = f"""
You are an expert Named Entity Recognition (NER) system.

Below are few-shot examples.

{fewshot_context}

TASK:
Identify the correct entity type for the candidate entity.

ENTITY TYPES:
- PER
- ORG
- LOC
- MISC

CURRENT SENTENCE:
"{sentence}"

CANDIDATE ENTITY:
"{entity}"

METRIC-BASED SLM PREDICTION:
"{metric_label}"

BERT NER PREDICTION:
"{bert_label}"

BERT CONFIDENCE:
"{bert_confidence}"

INSTRUCTIONS:
1. Analyze semantic meaning carefully.
2. Use the few-shot examples as guidance.
3. Return ONLY the correct label.
4. If metric retrieval and BERT disagree,
carefully analyze semantic context.
5. Sports-related actions usually indicate PER entities.
6. Carefully distinguish between products, foods,
common nouns, and organizations.

Examples:
- "Apple released a new iPhone" → ORG
- "He ate an apple" → MISC or not an entity
- "Amazon hired engineers" → ORG
- "The amazon rainforest" → LOC
OUTPUT RULES:
- Return ONLY ONE WORD
- No explanation
- No reasoning
- No sentences
- Allowed outputs ONLY:
PER
ORG
LOC
MISC
"""

    try:
        # Call Groq LLM
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=5
        )

        # Raw LLM Output
        response_text = (
            chat_completion.choices[0]
            .message.content
            .strip()
            .upper()
        )

        # Extract Valid Label
        valid_labels = ["PER", "ORG", "LOC", "MISC"]
        corrected_label = None
        for label in valid_labels:
            if label in response_text:
                corrected_label = label
                break

        # Normalize Common Variants
        normalization_map = {
            "ORGANIZATION": "ORG",
            "LOCATION": "LOC",
            "PERSON": "PER",
            "MISCELLANEOUS": "MISC"
        }
        corrected_label = normalization_map.get(
            corrected_label,
            corrected_label
        )

        # Fallback Protection
        if corrected_label is None:
            corrected_label = predicted_label

        return corrected_label

    except Exception as e:
        print(f"[LLM Error] {e}")
        return None


# -----------------------------
# Test directly
# -----------------------------

if __name__ == "__main__":

    fewshot_examples = [
        {
            "sentence": "Messi won the football tournament.",
            "entity": "Messi",
            "label": "PER"
        },

        {
            "sentence": "Ronaldo scored a hat-trick yesterday.",
            "entity": "Ronaldo",
            "label": "PER"
        }
    ]

    sentence = "Jordan scored 40 points in the match."

    entity = "Jordan"

    predicted_label = "ORG"

    corrected = correct_label(
        sentence,
        entity,
        predicted_label,
        fewshot_examples
    )

    print("\nCorrected Label:", corrected)
