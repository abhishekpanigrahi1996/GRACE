# Qwen3.5-27B/

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from vllm import LLM, SamplingParams


def load_items(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix == ".jsonl":
        items = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    elif p.suffix == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("JSON file must contain a list of objects.")
    else:
        raise ValueError("Input file must be .json or .jsonl")


def save_items(path: str, items: List[Dict[str, Any]]) -> None:
    p = Path(path)
    if p.suffix == ".jsonl":
        with open(p, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    elif p.suffix == ".json":
        with open(p, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError("Output file must be .json or .jsonl")


def extract_prompt(item: Dict[str, Any], prompt_key: str) -> str:
    if prompt_key in item:
        return str(item[prompt_key])

    # fallback keys
    for key in ["question", "prompt", "input", "query", "setup"]:
        if key in item:
            return str(item[key])
    
    raise KeyError(f"Could not find prompt field in item. Tried '{prompt_key}', question, prompt, input, query.")


def build_messages(question: str, system_prompt: str = None) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    question = "Please reason step by step and put your final answer within \\boxed{{}}.\n\n" + question 
    messages.append({"role": "user", "content": question})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Local path to Qwen model")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_key", type=str, default="question",
                        help="Field name containing the question/prompt")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--responses_per_question", type=int, default=16)
    args = parser.parse_args()

    items = load_items(args.input_file)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    tokenizer = llm.get_tokenizer()

    prompts = []
    for item in items:
        question = extract_prompt(item, args.prompt_key)
        messages = build_messages(question, args.system_prompt)

        # Qwen chat formatting
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        prompts.append(prompt_text)

    sampling_params = SamplingParams(
        n=args.responses_per_question,   
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for item, output in zip(items, outputs):
        item_copy = dict(item)
        item_copy["responses"] = [cand.text.strip() for cand in output.outputs]
        results.append(item_copy)

    save_items(args.output_file, results)
    print(f"Saved {len(results)} items to {args.output_file}")


if __name__ == "__main__":
    main()