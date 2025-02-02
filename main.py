import os
import json
from arg_parser import parse_args
from cper_framework import CPERFramework

def main():
    args = parse_args()

    # Define paths for the prompt files (ensure the "prompts" folder is in your project root)
    extract_persona_path = os.path.join("prompts", "extract_persona.txt")
    feedback_path = os.path.join("prompts", "feedback_and_action.txt")
    retrieve_persona_path = os.path.join("prompts", "retrive_persona.txt")
    refine_response_path = os.path.join("prompts", "refine_response.txt")

    # Initialize the CPER framework with the provided prompt files and parameters
    cper = CPERFramework(
        extract_persona_file=extract_persona_path,
        feedback_file=feedback_path,
        retrieve_persona_file=retrieve_persona_path,
        refine_response_file=refine_response_path,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens
    )

    # Load the dataset (assumed to be a JSON file)
    with open(args.dataset, 'r', encoding='utf-8') as dataset_file:
        data = json.load(dataset_file)

    # Process each conversation in the dataset.
    # For CCPE: each conversation contains a "conversation_id" and "turns" (each with "user_input" and "response").
    # For ESConv: additional fields exist but we focus on "turns".
    for conversation in data:
        conv_id = conversation.get("conversation_id", "unknown")
        print("=" * 100)
        print(f"Processing Conversation ID: {conv_id}")
        cper.reset_conversation()  # clear history for each new conversation

        turns = conversation.get("turns", [])
        for turn in turns:
            user_input = turn.get("user_input", "").strip()
            if not user_input:
                continue
            print("User:", user_input)
            response = cper.process_turn(user_input)
            print("CPER Response:", response)
            print("-" * 100)

if __name__ == "__main__":
    main()
