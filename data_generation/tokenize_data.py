import argparse
import json
import os
from datasets import Dataset
from typing import Any, Dict, List
import random
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create tokenized supervision data from problem/response pairs."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input .json or .jsonl file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output hf file.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Tokenizer name or local path.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional max length for truncation.",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        default=-100,
        help="Label value for non-solution tokens. Use -100 for HF training.",
    )
    parser.add_argument(
        "--problem-key",
        type=str,
        default="problem",
        help="Key for problem text.",
    )
    parser.add_argument(
        "--responses-key",
        type=str,
        default=None,
        help="Optional explicit key for list of responses, e.g. responses or solutions.",
    )
    parser.add_argument(
        "--response-key",
        type=str,
        default=None,
        help="Optional explicit key for single response, e.g. solution or response.",
    )

    parser.add_argument(
        "--subsample-questions",
        type=int,
        default=-1,
        help="Optional - if you want to subsample few questions",
    )

    parser.add_argument(
        "--subsample-responses",
        type=int,
        default=-1,
        help="Optional - if you want to subsample few responses per question",
    )
    return parser.parse_args()


def load_data(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        raise ValueError("JSON file must contain a list of examples.")

    raise ValueError("Input file must end with .json or .jsonl")


def extract_prompt(item: Dict[str, Any], prompt_key: str) -> str:
    if prompt_key in item:
        return str(item[prompt_key])

    # fallback keys
    for key in ["question", "prompt", "input", "query", "setup"]:
        if key in item:
            return str(item[key])
    
    raise KeyError(f"Could not find prompt field in item. Tried '{prompt_key}', question, prompt, input, query.")



def find_responses(example: Dict[str, Any], responses_key: str = None, response_key: str = None) -> List[str]:
    if responses_key is not None:
        value = example[responses_key]
        if not isinstance(value, list):
            raise ValueError(f"Expected list at key '{responses_key}', got {type(value)}")
        return [str(x) for x in value]

    if response_key is not None:
        return [str(example[response_key])]

    for key in ["responses", "solutions", "completions", "outputs"]:
        if key in example:
            value = example[key]
            if not isinstance(value, list):
                raise ValueError(f"Expected list at key '{key}', got {type(value)}")
            return [str(x) for x in value]

    raise KeyError(
        "Could not find responses. Provide --responses-key or --response-key, "
        "or use one of: responses, solutions, completions, outputs, solution, response, completion, output."
    )


def build_example(problem: str, solution: str, tokenizer, ignore_index: int, max_length: int = None) -> Dict[str, Any]:
    eos_text = tokenizer.eos_token if tokenizer.eos_token is not None else ""
    prompt_text = f"### Problem:\n{problem}\n\n### Solution:\n"
    full_text = prompt_text + solution + eos_text

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    labels = [ignore_index] * len(prompt_ids) + full_ids[len(prompt_ids):]

    if len(labels) != len(full_ids):
        raise RuntimeError("labels and input_ids length mismatch")

    if max_length is not None:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]

    return {
        "problem": problem,
        "solution": solution,
        "input_ids": full_ids,
        "labels": labels,
    }


def main():
    args = parse_args()

    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"Loading data from: {args.input_path}")
    data = load_data(args.input_path)
    print(f"Loaded {len(data)} question entries")

    processed = []

    if args.subsample_questions > 0:
        rng = random.Random(42)
        data = data[:]  
        rng.shuffle(data)
        data = data[:args.subsample_responses]

    for example_idx, example in enumerate(data):

        problem = extract_prompt(example, args.problem_key)
        responses = find_responses(
            example,
            responses_key=args.responses_key,
            response_key=args.response_key,
        )
        if args.subsample_responses > 0:
            rng = random.Random(example_idx * 13 + 13)
            responses = responses[:]  # avoid mutating original list
            rng.shuffle(responses)
            responses = responses[:args.subsample_responses]

        for response_idx, solution in enumerate(responses):
            item = build_example(
                problem=problem,
                solution=solution,
                tokenizer=tokenizer,
                ignore_index=args.ignore_index,
                max_length=args.max_length,
            )
            item["question_index"] = example_idx
            item["response_index"] = response_idx
            processed.append(item)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    
    
    dataset = Dataset.from_list(processed)
    dataset.save_to_disk(args.output_path)
    print(f"Saved {len(processed)} examples to: {args.output_path}")

    if processed:
        print("First example stats:")
        print(f"  input length: {len(processed[0]['input_ids'])}")
        print(f"  num supervised tokens: {sum(x != args.ignore_index for x in processed[0]['labels'])}")


if __name__ == "__main__":
    main()