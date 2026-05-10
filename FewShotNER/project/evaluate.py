import time
import main


# ===================================
# TEST CASES
# ===================================

test_cases = [

    {
        "sentence":
        "Elon Musk went to India for Tesla event.",

        "expected": {
            "Elon Musk": "PER",
            "India": "LOC",
            "Tesla": "ORG"
        }
    },

    {
        "sentence":
        "Riya went to Japan",

        "expected": {
            "Riya": "PER",
            "Japan": "LOC"
        }
    },

    {
        "sentence":
        "Ankit works at Amazon",

        "expected": {
            "Ankit": "PER",
            "Amazon": "ORG"
        }
    },

    {
        "sentence":
        "She Loves French Cuisine",

        "expected": {
            "French": "MISC"
        }
    },

    {
        "sentence":
        "The company launched new Iphone in Australia by Apple.",

        "expected": {
            "Iphone": "MISC",
            "Australia": "LOC",
            "Apple": "ORG"
        }
    },

    {
        "sentence":
        "Indian scientists worked on NASA missions.",

        "expected": {
            "Indian": "MISC",
            "NASA": "ORG"
        }
    },

    {
        "sentence":
        "He likes to eat Apple.",

        "expected": {
            "Apple": "O"
        }
    },

    {
        "sentence":
        "Iphone is owned by Apple.",

        "expected": {
            "Iphone": "MISC",
            "Apple": "ORG"
        }
    },

    {
        "sentence":
        "Global economies were affected by COVID-19.",

        "expected": {
            "COVID": "MISC"
        }
    },

    {
        "sentence":
        "Covid took so many lives",

        "expected": {
            "Covid": "MISC"
        }
    },

    {
        "sentence":
        "Priya visited Jaipur last week.",

        "expected": {
            "Priya": "PER",
            "Jaipur": "LOC"
        }
    },

    {
        "sentence":
        "Rahul studies at IIT Bombay.",

        "expected": {
            "Rahul": "PER",
            "IIT Bombay": "ORG"
        }
    },

    {
        "sentence":
        "ISRO launched a satellite successfully.",

        "expected": {
            "ISRO": "ORG"
        }
    },

    {
        "sentence":
        "Neha attended a conference at Infosys Bangalore campus.",

        "expected": {
            "Neha": "PER",
            "Infosys": "ORG",
            "Bangalore": "LOC"
        }
    }
]


# ===================================
# START TIMER
# ===================================

start_time = time.time()

correct_predictions = 0

wrong_predictions = 0

total_predictions = 0

print("\n==============================")
print("RUNNING EVALUATION")
print("==============================\n")


# ===================================
# RUN TESTS
# ===================================

for test in test_cases:

    sentence = test["sentence"]

    expected_entities = test["expected"]

    print("\n--------------------------------")
    print(f"Sentence: {sentence}")
    print("--------------------------------")

#     predictions = main.process_sentence(
#     sentence,
#     use_detector=False,
#     use_memory=False,
#     simple_threshold_mode=True
# )
    predictions = main.process_sentence(
    sentence
)

    # -----------------------------------
    # Normalize predictions
    # -----------------------------------

    predicted_map = {}

    for pred in predictions:

        normalized_entity = (
            pred["entity"]
            .strip()
            .lower()
        )

        predicted_map[
            normalized_entity
        ] = pred["final_label"]

    # -----------------------------------
    # Compare predictions
    # -----------------------------------

    for entity, true_label in expected_entities.items():

        total_predictions += 1

        normalized_entity = (
            entity
            .strip()
            .lower()
        )

        predicted_label = predicted_map.get(
            normalized_entity,
            "NOT_FOUND"
        )

        # -----------------------------------
        # Correct prediction
        # -----------------------------------

        if predicted_label == true_label:

            correct_predictions += 1

            print(
                f"[CORRECT] "
                f"{entity} "
                f"→ "
                f"{predicted_label}"
            )

        # -----------------------------------
        # Wrong prediction
        # -----------------------------------

        else:

            wrong_predictions += 1

            print(
                f"[WRONG] "
                f"{entity}"
            )

            print(
                f"Expected: "
                f"{true_label}"
            )

            print(
                f"Predicted: "
                f"{predicted_label}"
            )


# ===================================
# FINAL METRICS
# ===================================

end_time = time.time()

accuracy = (
    correct_predictions /
    total_predictions
) * 100

execution_time = round(
    end_time - start_time,
    2
)

print("\n==============================")
print("FINAL EVALUATION")
print("==============================")

print(
    f"\nAccuracy: "
    f"{accuracy:.2f}%"
)

print(
    f"Correct Predictions: "
    f"{correct_predictions}"
)

print(
    f"Wrong Predictions: "
    f"{wrong_predictions}"
)

print(
    f"Total Predictions: "
    f"{total_predictions}"
)

print(
    f"LLM Calls Made: "
    f"{main.llm_call_count}"
)

print(
    f"Execution Time: "
    f"{execution_time} seconds"
)