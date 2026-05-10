# memory.py

import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Memory file
MEMORY_FILE = "memory.json"

# Embedding model
model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

# Load memory
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as file:
        return json.load(file)

# Save memory
def save_memory(memory_data):
    with open(MEMORY_FILE, "w") as file:
        json.dump(
            memory_data,
            file,
            indent=4
        )

# Save correction
def save_correction(sentence, entity, wrong_label, correct_label):
    memory = load_memory()
    new_entry = {
        "sentence": sentence,
        "entity": entity,
        "wrong_label": wrong_label,
        "correct_label": correct_label
    }
    memory.append(new_entry)
    save_memory(memory)
    print(
        f"[Memory] Saved correction "
        f"for '{entity}'"
    )

# Context-aware memory retrieval
def retrieve_correction(sentence, entity, threshold=0.5):
    memory = load_memory()

    # No memory yet
    if len(memory) == 0:
        return None

    # Query embedding
    query_text = (

        f"Entity: {entity}. "

        f"Sentence: {sentence}"
    )
    query_embedding = model.encode([query_text])
    best_similarity = 0
    best_label = None

    # Search memory
    for entry in memory:

        # Entity names must match
        if entry["entity"].lower() != entity.lower():

            continue

        memory_text = (
            f"Entity: {entry['entity']}. "

            f"Sentence: {entry['sentence']}"
        )

        memory_embedding = model.encode([memory_text])
        similarity = cosine_similarity(
            query_embedding,
            memory_embedding
        )[0][0]

        # Best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_label = entry["correct_label"]
        print("Similarity: ", similarity)

    # Threshold check
    if best_similarity >= threshold:
        print(

            f"[Memory] Similar correction found "

            f"(Similarity: "

            f"{round(best_similarity, 4)})"

        )
        return best_label
    return None

# -----------------------------------
# Test directly
# -----------------------------------

if __name__ == "__main__":

    save_correction(

        sentence="Jordan scored 40 points",

        entity="Jordan",

        wrong_label="ORG",

        correct_label="PER"
    )

    result = retrieve_correction(

        sentence="Jordan won the basketball match",

        entity="Jordan"
    )

    print("\nRetrieved Label:", result)