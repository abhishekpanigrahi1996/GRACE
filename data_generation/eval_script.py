import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np

from math_parsing_utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute average math accuracy across multiple responses per question."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input JSON/JSONL/PKL file containing problems, answers, and 16 responses.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to save detailed results as pickle.",
    )
    parser.add_argument(
        "--problem-key",
        type=str,
        default="problem",
        help="Key for problem text.",
    )
    parser.add_argument(
        "--answer-key",
        type=str,
        default="answer",
        help="Key for gold/reference answer.",
    )
    parser.add_argument(
        "--responses-key",
        type=str,
        default="responses",
        help="Key for list of responses.",
    )
    parser.add_argument(
        "--expected-num-responses",
        type=int,
        default=16,
        help="Expected number of responses per question.",
    )
    parser.add_argument(
        "--strict-num-responses",
        action="store_true",
        help="If set, raise an error when an example does not have expected-num-responses responses.",
    )
    parser.add_argument(
        "--use-last-number",
        action="store_true",
        help="Pass use_last_number=True to extract_answer.",
    )
    parser.add_argument(
        "--include-percentage",
        action="store_true",
        help="Allow percentage equivalence in math_equal.",
    )
    parser.add_argument(
        "--no-is-close",
        action="store_true",
        help="Disable approximate numeric comparison in math_equal.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="math",
        choices=["math", "multiple_choice", "mmlu_pro"],
        help="How to extract predicted answers from raw responses.",
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
        if not isinstance(obj, list):
            raise ValueError("JSON file must contain a list of examples.")
        return obj

    if path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, list):
            raise ValueError("Pickle file must contain a list of examples.")
        return obj

    raise ValueError("Unsupported file format. Use .json, .jsonl, or .pkl")


def extract_prediction(text: str, dataset_type: str, use_last_number: bool) -> str:
    if dataset_type == "math":
        return extract_answer(text, use_last_number=use_last_number)
    if dataset_type == "multiple_choice":
        return get_multiple_choice_answer(text)
    if dataset_type == "mmlu_pro":
        pred = mmlu_pro_extract_answer(text)
        return "" if pred is None else pred
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def get_reference_answer(example: Dict[str, Any], answer_key: str) -> str:
    if answer_key not in example:
        raise KeyError(f"Missing answer key '{answer_key}'")
    return str(example[answer_key])


def get_responses(example: Dict[str, Any], responses_key: str) -> List[str]:
    if responses_key not in example:
        raise KeyError(f"Missing responses key '{responses_key}'")
    responses = example[responses_key]
    if not isinstance(responses, list):
        raise ValueError(f"Expected '{responses_key}' to be a list, got {type(responses)}")
    return [str(x) for x in responses]


def evaluate_example(
    example: Dict[str, Any],
    answer_key: str,
    responses_key: str,
    dataset_type: str,
    use_last_number: bool,
    include_percentage: bool,
    is_close: bool,
) -> Dict[str, Any]:
    reference = get_reference_answer(example, answer_key)
    reference = extract_prediction(text=reference, dataset_type=dataset_type, use_last_number=use_last_number,)
    responses = get_responses(example, responses_key)

    extracted_predictions = []
    correctness = []

    for response in responses:
        pred = extract_prediction(
            text=response,
            dataset_type=dataset_type,
            use_last_number=use_last_number,
        )
        correct = math_equal(
            prediction=pred,
            reference=reference,
            include_percentage=include_percentage,
            is_close=is_close,
        )
        extracted_predictions.append(pred)
        correctness.append(bool(correct))

    correctness = np.asarray(correctness, dtype=np.float32)

    return {
        "reference": reference,
        "num_responses": len(responses),
        "raw_responses": responses,
        "parsed_predictions": extracted_predictions,
        "correctness": correctness.tolist(),
        "avg_accuracy": float(correctness.mean()) if len(correctness) > 0 else 0.0,
        "pass_at_1": float(correctness[0]) if len(correctness) > 0 else 0.0,
        "pass_at_k": float(correctness.max()) if len(correctness) > 0 else 0.0,
        "num_correct": int(correctness.sum()),
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(results) == 0:
        return {
            "num_examples": 0,
            "avg_accuracy_across_16": 0.0,
            "avg_pass_at_1": 0.0,
            "avg_pass_at_k": 0.0,
            "avg_num_correct": 0.0,
        }

    avg_accuracy = float(np.mean([r["avg_accuracy"] for r in results]))
    avg_pass_at_1 = float(np.mean([r["pass_at_1"] for r in results]))
    avg_pass_at_k = float(np.mean([r["pass_at_k"] for r in results]))
    avg_num_correct = float(np.mean([r["num_correct"] for r in results]))

    return {
        "num_examples": len(results),
        "avg_accuracy_across_16": avg_accuracy,
        "avg_pass_at_1": avg_pass_at_1,
        "avg_pass_at_k": avg_pass_at_k,
        "avg_num_correct": avg_num_correct,
    }


def main():
    args = parse_args()

    print(f"Loading data from: {args.input_path}")
    data = load_data(args.input_path)
    print(f"Loaded {len(data)} examples")

    detailed_results = []

    for idx, example in enumerate(data):
        result = evaluate_example(
            example=example,
            answer_key=args.answer_key,
            responses_key=args.responses_key,
            dataset_type=args.dataset_type,
            use_last_number=args.use_last_number,
            include_percentage=args.include_percentage,
            is_close=not args.no_is_close,
        )

        if args.strict_num_responses and result["num_responses"] != args.expected_num_responses:
            raise ValueError(
                f"Example {idx} has {result['num_responses']} responses, "
                f"expected {args.expected_num_responses}"
            )
        detailed_results += [result]

    summary = summarize_results(detailed_results)

    print("\n===== Summary =====")
    print(f"Num examples            : {summary['num_examples']}")
    print(f"Avg accuracy across 16  : {summary['avg_accuracy_across_16']:.6f}")
    print(f"Avg pass@1              : {summary['avg_pass_at_1']:.6f}")
    print(f"Avg pass@k              : {summary['avg_pass_at_k']:.6f}")
    print(f"Avg #correct / question : {summary['avg_num_correct']:.6f}")



if __name__ == "__main__":
    main()