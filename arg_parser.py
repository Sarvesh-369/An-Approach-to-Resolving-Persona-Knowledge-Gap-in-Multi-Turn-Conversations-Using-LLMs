import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run CPER Framework on a given dataset")
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="Path to the JSON dataset file (CCPE or ESConv format)"
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        choices=["ccpe", "esconv"],
        default="ccpe",
        help="Type of dataset: 'ccpe' or 'esconv' (default: ccpe)"
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help="Temperature for text generation (default: 0.7)"
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=400,
        help="Maximum number of new tokens for generated responses (default: 400)"
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='output.txt',
        help="File to log outputs (default: output.txt)"
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print("Parsed arguments:")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Output file: {args.output_file}")
