import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# Load support examples
with open("data/support.json", "r") as file:
    support_examples = json.load(file)

# Create embeddings
support_sentences = [
    f"Entity: {item['entity']}. "
    f"Sentence: {item['sentence']}"
    for item in support_examples
]
support_embeddings = model.encode(
    support_sentences
)

# Metric-Based Few-Shot Prediction
def retrieve_similar_examples(query_sentence,entity,top_k=3,filter_majority=True):
    # Create entity-aware query
    query_text = (

    f"Entity: {entity}. "

    f"Sentence: {query_sentence}"
)

    # Encode query
    query_embedding = model.encode([query_text])

    # Cosine similarity
    similarities = cosine_similarity(query_embedding, support_embeddings)[0]

    # Top-k retrieval
    top_indices = similarities.argsort()[-top_k:][::-1]
    retrieved_examples = []
    label_scores = {}
    similarity_sum = 0

    # Retrieve examples
    for idx in top_indices:
        example = support_examples[idx].copy()
        similarity_score = round(float(similarities[idx]), 4)
        example["similarity_score"] = similarity_score
        retrieved_examples.append(example)

        # Weighted label voting
        label = example["label"]
        label_scores[label] = (
            label_scores.get(label, 0)
            + similarity_score
        )
        similarity_sum += similarity_score

    # Final predicted label
    predicted_label = max(label_scores,key=label_scores.get)

    if filter_majority:
        filtered_examples = [
        ex for ex in retrieved_examples
        if ex["label"] == predicted_label
    ]
    else:
        filtered_examples = retrieved_examples

# Better confidence
    confidence = round(label_scores[predicted_label]/similarity_sum, 4)

# Return results
    return {
    "predicted_label":
        predicted_label,
    "confidence":
        confidence,
    "retrieved_examples":
        filtered_examples
    }

# -----------------------------------
# Test directly
# -----------------------------------
if __name__ == "__main__":

    query_sentence = (
        "Jordan is in California"
    )

    entity = "Jordan"

    result = retrieve_similar_examples(
        query_sentence,
        entity
    )

    print("\nSLM Prediction:\n")

    print(
        "Predicted Label:",
        result["predicted_label"]
    )

    print(
        "Confidence:",
        result["confidence"]
    )

    print("\nRetrieved Examples:\n")

    for example in result["retrieved_examples"]:
        print(example)