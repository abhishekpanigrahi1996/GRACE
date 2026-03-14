import argparse
import json
import os
import pickle

import torch
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from trak.projectors import CudaProjector, ProjectionType


projectorcls = CudaProjector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-example gradients (optionally LoRA-only), project them with a JL projector, and save them."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/Llama-3.2-3B-Instruct",
        help="Path to the student model.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to input hf data file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/projected_gradients_Qwen-3B.pkl",
        help="Path to save projected gradients.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on, e.g. cuda or cpu.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="If set, wrap the model with LoRA and collect LoRA gradients only.",
    )
    parser.add_argument(
        "--proj-dim",
        type=int,
        default=8192,
        help="Projection dimension for JL gradient projection.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model loading dtype.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "auto":
        return "auto"
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def load_jsonl_as_dataset(path: str) -> Dataset:
    all_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            all_data.append(json.loads(line))
    return Dataset.from_list(all_data)


def get_grad_dim(model: torch.nn.Module, use_lora: bool) -> int:
    total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if use_lora:
            if "lora_B" in name:
                total += param.numel()
        else:
            total += param.numel()
    return total


def build_model(student_pth: str, device: str, use_lora: bool, proj_dim: int, dtype: str):
    torch_dtype = get_torch_dtype(dtype)

    if torch_dtype == "auto":
        model = AutoModelForCausalLM.from_pretrained(student_pth).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            student_pth, torch_dtype=torch_dtype
        ).to(device)

    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    grad_dim = get_grad_dim(model, use_lora)
    if grad_dim == 0:
        raise RuntimeError("Gradient dimension is zero. Check trainable parameters / LoRA setup.")

    model_device = next(model.parameters()).device

    gradientproj = projectorcls(
        proj_dim=int(proj_dim),
        grad_dim=grad_dim,
        seed=0,
        proj_type=ProjectionType.rademacher,
        device=model_device,
        dtype=torch.bfloat16,
        block_size=4,
        max_batch_size=8,
    )

    normalize_factor = torch.sqrt(
        torch.tensor(float(grad_dim), dtype=torch.float32, device=model_device)
    )

    return model, gradientproj, normalize_factor, grad_dim


def extract_gradients(model: torch.nn.Module, use_lora: bool) -> torch.Tensor:
    gradients = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            continue

        if use_lora:
            if "lora_B" in name:
                gradients.append(param.grad.reshape(-1))
        else:
            gradients.append(param.grad.reshape(-1))

    if not gradients:
        if use_lora:
            raise RuntimeError("No gradients found for LoRA parameters.")
        raise RuntimeError("No gradients found for trainable model parameters.")

    return torch.cat(gradients, dim=0)



def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from: {args.model}")
    print(f"Using LoRA: {args.use_lora}")
    print(f"Projection dim: {args.proj_dim}")

    model, gradientproj, normalize_factor, grad_dim = build_model(
        student_pth=args.model,
        device=args.device,
        use_lora=args.use_lora,
        proj_dim=args.proj_dim,
        dtype=args.dtype,
    )

    device = next(model.parameters()).device
    print(f"Gradient dimension before projection: {grad_dim}")

    print(f"Loading data from: {args.data_path}")
    data = load_from_disk(args.data_path)
    print(f"Processing {len(data)} examples")

    all_projected_grads = []

    for d in tqdm(data):
        input_ids = torch.tensor(d["input_ids"], device=device).unsqueeze(0)
        labels = torch.tensor(d["labels"], device=device).unsqueeze(0)

       
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        grads = extract_gradients(model, args.use_lora)
        collected_grads = grads.unsqueeze(0)
        projected_grad = gradientproj.project(collected_grads, model_id=0) / normalize_factor

        all_projected_grads.append(projected_grad.squeeze(0).detach().cpu().numpy())

        del grads, collected_grads, projected_grad, outputs, loss, input_ids, labels
        model.zero_grad(set_to_none=True)

    print(f"Saving projected gradients to: {args.output_path}")
    with open(args.output_path, "wb") as f:
        pickle.dump(all_projected_grads, f)

    print("Done.")


if __name__ == "__main__":
    main()