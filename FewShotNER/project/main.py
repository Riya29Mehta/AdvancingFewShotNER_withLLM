
from detector import is_suspicious
from llm_corrector import correct_label
from memory import save_correction, retrieve_correction
from fewshot_retreiver import retrieve_similar_examples
from ner_model import predict_entities

llm_call_count = 0
def process_sentence(sentence):

    print("\n==============================")
    print("INPUT SENTENCE:")
    print(sentence)
    print("==============================\n")

    # -----------------------------------
    # STEP 1: Detect entities using BERT
    # -----------------------------------

    bert_entities = predict_entities(sentence)

    if not bert_entities:
        print("\nNo entities detected.")
        return

    print("\n[Detected Entities]\n")

    for item in bert_entities:
        print(f"{item['entity']}")

    # -----------------------------------
    # STEP 2: Process each entity
    # -----------------------------------

    final_results = []

    for item in bert_entities:

        entity = item["entity"]

        bert_label = item["label"]

        bert_confidence = item["confidence"]

        print("\n===================================")
        print(f"PROCESSING ENTITY: {entity}")
        print("===================================\n")

        # =====================================================
        # STEP 3: MEMORY CHECK FIRST
        # =====================================================

        memory_label = retrieve_correction(
            sentence,
            entity
        )

        if memory_label:

            print("\n[Memory Lookup]")
            print(
                f"Stored correction found for "
                f"'{entity}'"
            )

            print(
                f"{entity} → {memory_label}"
            )

            final_label = memory_label

        else:

            print(
                "\n[Memory Lookup]"
            )

            print(
                f"No memory found for "
                f"'{entity}'"
            )

            # =====================================================
            # STEP 4: Metric-Based SLM Prediction
            # =====================================================

            slm_result = retrieve_similar_examples(
                sentence,
                entity,
                filter_majority=True
            )

            predicted_label = (
                slm_result["predicted_label"]
            )

            confidence = (
                slm_result["confidence"]
            )

            fewshot_examples = (
                slm_result["retrieved_examples"]
            )

            print(
                "\n[Metric-Based SLM Prediction]"
            )

            print(
                f"Entity: {entity}"
            )

            print(
                f"Predicted Label: "
                f"{predicted_label}"
            )

            print(
                f"Confidence: "
                f"{confidence}"
            )

            # =====================================================
            # STEP 5: BERT Verification
            # =====================================================

            print("\n[BERT Verification]")

            print(
                f"BERT Label: "
                f"{bert_label}"
            )

            print(
                f"BERT Confidence: "
                f"{bert_confidence}"
            )

            # =====================================================
            # STEP 6: Suspicious Detection
            # =====================================================

            suspicious = is_suspicious(
                predicted_label,
                bert_label,
                confidence,
                bert_confidence
            )

            print("\n[Detector Output]")

            print(
                f"Suspicious Status: "
                f"{suspicious}"
            )

            print(
                f"SLM Confidence: "
                f"{confidence}"
            )

            print(
                f"BERT Confidence: "
                f"{bert_confidence}"
            )

            # =====================================================
            # STEP 7: LLM Correction
            # =====================================================

            if suspicious != "Not_sus":

                print(
                    f"\n[Detector] "
                    f"'{entity}' marked suspicious"
                )

                expanded_examples = []

                # -----------------------------------
                # High confidence mismatch
                # -----------------------------------

                if suspicious == "High_conf":

                    print(
                        "\n[Adaptive Retrieval]"
                    )

                    print(
                        "Using expanded "
                        "top-10 examples for LLM"
                    )

                    expanded_result = (
                        retrieve_similar_examples(
                            sentence,
                            entity,
                            top_k=10,
                            filter_majority=False
                        )
                    )

                    expanded_examples = (
                        expanded_result[
                            "retrieved_examples"
                        ]
                    )

                # -----------------------------------
                # Low confidence case
                # -----------------------------------

                elif suspicious == "Low_conf":

                    print(
                        "\n[Adaptive Retrieval]"
                    )

                    print(
                        "Using existing "
                        "few-shot examples"
                    )

                    expanded_examples = (
                        fewshot_examples
                    )

                # -----------------------------------
                # LLM correction
                # -----------------------------------
                global llm_call_count
                llm_call_count += 1
                corrected_label = correct_label(
                    sentence,
                    entity,
                    predicted_label,
                    bert_label,
                    bert_confidence,
                    expanded_examples
                )

                print(
                    f"\n[LLM Correction] "
                    f"{entity} → "
                    f"{corrected_label}"
                )

                # -----------------------------------
                # Save to memory
                # -----------------------------------

                save_correction(
                    sentence,
                    entity,
                    predicted_label,
                    corrected_label
                )

                print(
                    f"[Memory] Saved correction "
                    f"for '{entity}'"
                )

                final_label = corrected_label

            # =====================================================
            # STEP 8: Non-suspicious entity
            # =====================================================

            else:

                print(
                    f"\n[Detector] "
                    f"'{entity}' is NOT suspicious"
                )

                final_label = predicted_label

        # =====================================================
        # STEP 9: Store final result
        # =====================================================

        final_results.append({

            "entity": entity,

            "final_label": final_label,

            "confidence": (
                bert_confidence
            )
        })

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    for result in final_results:

        print(
            f"{result['entity']} "
            f"→ "
            f"{result['final_label']} "
            f"(Confidence: "
            f"{result['confidence']})"
        )

    return final_results

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    c = 1

    while c == 1:

        sentence = input(
            "\nEnter a sentence:\n\n"
        )

        process_sentence(sentence)

        c = int(
            input(
                "\nContinue? "
                "1: Yes | 0: No --> "
            )
        )





















# from detector import is_suspicious
# from llm_corrector import correct_label
# from memory import save_correction, retrieve_correction
# from fewshot_retreiver import retrieve_similar_examples
# from ner_model import predict_entities


# # =========================================
# # GLOBAL COUNTER
# # =========================================

# llm_call_count = 0


# # =========================================
# # MAIN PROCESS FUNCTION
# # =========================================

# def process_sentence(
#     sentence,
#     use_detector=True,
#     use_memory=True,
#     simple_threshold_mode=False
# ):

#     print("\n==============================")
#     print("INPUT SENTENCE:")
#     print(sentence)
#     print("==============================\n")

#     # -----------------------------------
#     # STEP 1: Detect entities using BERT
#     # -----------------------------------

#     bert_entities = predict_entities(sentence)

#     if not bert_entities:

#         print("\nNo entities detected.")

#         return []

#     print("\n[Detected Entities]\n")

#     for item in bert_entities:

#         print(f"{item['entity']}")

#     # -----------------------------------
#     # STEP 2: Process each entity
#     # -----------------------------------

#     final_results = []

#     for item in bert_entities:

#         entity = item["entity"]

#         bert_label = item["label"]

#         bert_confidence = item["confidence"]

#         print("\n===================================")

#         print(f"PROCESSING ENTITY: {entity}")

#         print("===================================\n")

#         # =====================================================
#         # STEP 3: MEMORY CHECK
#         # =====================================================

#         memory_label = None

#         if use_memory:

#             memory_label = retrieve_correction(
#                 sentence,
#                 entity
#             )

#         if memory_label:

#             print("\n[Memory Lookup]")

#             print(
#                 f"Stored correction found for "
#                 f"'{entity}'"
#             )

#             print(
#                 f"{entity} → {memory_label}"
#             )

#             final_label = memory_label

#         else:

#             print("\n[Memory Lookup]")

#             print(
#                 f"No memory found for "
#                 f"'{entity}'"
#             )

#             # =====================================================
#             # STEP 4: Metric-Based SLM Prediction
#             # =====================================================

#             slm_result = retrieve_similar_examples(
#                 sentence,
#                 entity,
#                 filter_majority=True
#             )

#             predicted_label = (
#                 slm_result["predicted_label"]
#             )

#             confidence = (
#                 slm_result["confidence"]
#             )

#             fewshot_examples = (
#                 slm_result["retrieved_examples"]
#             )

#             print(
#                 "\n[Metric-Based SLM Prediction]"
#             )

#             print(
#                 f"Entity: {entity}"
#             )

#             print(
#                 f"Predicted Label: "
#                 f"{predicted_label}"
#             )

#             print(
#                 f"Confidence: "
#                 f"{confidence}"
#             )

#             # =====================================================
#             # STEP 5: BERT Verification
#             # =====================================================

#             print("\n[BERT Verification]")

#             print(
#                 f"BERT Label: "
#                 f"{bert_label}"
#             )

#             print(
#                 f"BERT Confidence: "
#                 f"{bert_confidence}"
#             )

#             # =====================================================
#             # STEP 6: DETECTOR / THRESHOLD LOGIC
#             # =====================================================

#             # -----------------------------------
#             # SIMPLE THRESHOLD BASELINE
#             # -----------------------------------

#             if simple_threshold_mode:

#                 threshold = 0.68

#                 if confidence < threshold:

#                     suspicious = "Low_conf"

#                 else:

#                     suspicious = "Not_sus"

#             # -----------------------------------
#             # FULL DETECTOR MODE
#             # -----------------------------------

#             elif use_detector:

#                 suspicious = is_suspicious(
#                     predicted_label,
#                     bert_label,
#                     confidence,
#                     bert_confidence
#                 )

#             # -----------------------------------
#             # NO DETECTOR
#             # -----------------------------------

#             else:

#                 suspicious = "Not_sus"

#             print("\n[Detector Output]")

#             print(
#                 f"Suspicious Status: "
#                 f"{suspicious}"
#             )

#             print(
#                 f"SLM Confidence: "
#                 f"{confidence}"
#             )

#             print(
#                 f"BERT Confidence: "
#                 f"{bert_confidence}"
#             )

#             # =====================================================
#             # STEP 7: LLM Correction
#             # =====================================================

#             if suspicious != "Not_sus":

#                 print(
#                     f"\n[Detector] "
#                     f"'{entity}' marked suspicious"
#                 )

#                 expanded_examples = []

#                 # -----------------------------------
#                 # FULL MODEL:
#                 # adaptive retrieval
#                 # -----------------------------------

#                 if use_detector:

#                     # High confidence mismatch
#                     if suspicious == "High_conf":

#                         print(
#                             "\n[Adaptive Retrieval]"
#                         )

#                         print(
#                             "Using expanded "
#                             "top-10 examples for LLM"
#                         )

#                         expanded_result = (
#                             retrieve_similar_examples(
#                                 sentence,
#                                 entity,
#                                 top_k=10,
#                                 filter_majority=False
#                             )
#                         )

#                         expanded_examples = (
#                             expanded_result[
#                                 "retrieved_examples"
#                             ]
#                         )

#                     # Low confidence case
#                     elif suspicious == "Low_conf":

#                         print(
#                             "\n[Adaptive Retrieval]"
#                         )

#                         print(
#                             "Using existing "
#                             "few-shot examples"
#                         )

#                         expanded_examples = (
#                             fewshot_examples
#                         )

#                 # -----------------------------------
#                 # SIMPLE BASELINE:
#                 # only existing examples
#                 # -----------------------------------

#                 else:

#                     expanded_examples = (
#                         fewshot_examples
#                     )

#                 # -----------------------------------
#                 # LLM correction
#                 # -----------------------------------

#                 global llm_call_count

#                 llm_call_count += 1

#                 corrected_label = correct_label(
#                     sentence,
#                     entity,
#                     predicted_label,
#                     bert_label,
#                     bert_confidence,
#                     expanded_examples
#                 )

#                 print(
#                     f"\n[LLM Correction] "
#                     f"{entity} → "
#                     f"{corrected_label}"
#                 )

#                 # -----------------------------------
#                 # Save to memory
#                 # -----------------------------------

#                 if use_memory:

#                     save_correction(
#                         sentence,
#                         entity,
#                         predicted_label,
#                         corrected_label
#                     )

#                     print(
#                         f"[Memory] Saved correction "
#                         f"for '{entity}'"
#                     )

#                 final_label = corrected_label

#             # =====================================================
#             # STEP 8: Non-suspicious entity
#             # =====================================================

#             else:

#                 print(
#                     f"\n[Detector] "
#                     f"'{entity}' is NOT suspicious"
#                 )

#                 final_label = predicted_label

#         # =====================================================
#         # STEP 9: Store final result
#         # =====================================================

#         final_results.append({

#             "entity": entity,

#             "final_label": final_label,

#             "confidence": (
#                 bert_confidence
#             )
#         })

#     # =====================================================
#     # FINAL OUTPUT
#     # =====================================================

#     print("\n==============================")

#     print("FINAL RESULTS")

#     print("==============================")

#     for result in final_results:

#         print(
#             f"{result['entity']} "
#             f"→ "
#             f"{result['final_label']} "
#             f"(Confidence: "
#             f"{result['confidence']})"
#         )

#     return final_results


# # =====================================================
# # MAIN
# # =====================================================

# if __name__ == "__main__":

#     c = 1

#     while c == 1:

#         sentence = input(
#             "\nEnter a sentence:\n\n"
#         )

#         process_sentence(sentence)

#         c = int(
#             input(
#                 "\nContinue? "
#                 "1: Yes | 0: No --> "
#             )
#         )