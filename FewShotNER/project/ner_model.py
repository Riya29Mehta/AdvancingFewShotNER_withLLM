# from transformers import pipeline

# # Load pretrained NER pipeline
# ner_pipeline = pipeline(
#     "ner",
#     model="dslim/bert-base-NER",
#     aggregation_strategy="simple"
# )


# def predict_entities(sentence):
#     """
#     Predict named entities from a sentence.

#     Returns:
#     [
#         {
#             "entity": "Apple",
#             "label": "ORG",
#             "confidence": 0.99
#         }
#     ]
#     """

#     results = ner_pipeline(sentence)

#     entities = []

#     for result in results:
#         entity_data = {
#             "entity": result["word"],
#             "label": result["entity_group"],
#             "confidence": round(result["score"], 4)
#         }

#         entities.append(entity_data)

#     return entities


# # Test the file directly
# if __name__ == "__main__":

#     test_sentence = "Apple launched the new iPhone in California."

#     predictions = predict_entities(test_sentence)

#     print("\nDetected Entities:\n")

#     for pred in predictions:
#         print(pred)


from transformers import pipeline


# -----------------------------------
# Load pretrained BERT NER
# -----------------------------------

ner_pipeline = pipeline(

    "ner",

    model="dslim/bert-base-NER",

    aggregation_strategy="simple"
)


# -----------------------------------
# Clean tokenized words
# -----------------------------------

def clean_entity_text(text):

    # Remove BERT subword markers
    text = text.replace(" ##", "")

    text = text.replace("##", "")

    return text.strip()


# -----------------------------------
# Predict entities
# -----------------------------------

def predict_entities(sentence):

    results = ner_pipeline(sentence)

    entities = []

    for result in results:

        entity_text = clean_entity_text(
            result["word"]
        )

        entity_data = {

            "entity": entity_text,

            "label": result["entity_group"],

            "confidence": round(
                result["score"],
                4
            )
        }

        entities.append(entity_data)

    return entities


# -----------------------------------
# Test directly
# -----------------------------------

if __name__ == "__main__":

    test_sentence = (
        "Elon Musk visited India for Tesla event."
    )

    predictions = predict_entities(
        test_sentence
    )

    print("\nDetected Entities:\n")

    for pred in predictions:

        print(pred)


from transformers import pipeline


# -----------------------------------
# Load pretrained BERT NER
# -----------------------------------

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="first"
)


# -----------------------------------
# Predict entities
# -----------------------------------

def predict_entities(sentence):

    results = ner_pipeline(sentence)

    entities = []

    for result in results:

        entity_data = {
            "entity": result["word"],
            "label": result["entity_group"],
            "confidence": round(
                result["score"],
                4
            )
        }

        entities.append(entity_data)

    return entities


# -----------------------------------
# Test
# -----------------------------------

if __name__ == "__main__":

    test_sentences = [
        "Virat plays for India.",
        "Riya went to Japan",
        "Elon owns Tesla"
    ]

    for sentence in test_sentences:

        print("\n==============================")
        print("INPUT SENTENCE:")
        print(sentence)
        print("==============================")

        predictions = predict_entities(sentence)

        print("\n[Detected Entities]\n")

        for pred in predictions:
            print(pred)