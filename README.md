## In Good GRACES: Principled Teacher Selection for Knowledge Distillation

This repository contains the code for our paper [In Good GRACES: Principled Teacher Selection for Knowledge Distillation](https://openreview.net/forum?id=m276fke38H), accepted at [ICLR 2026](https://iclr.cc/). You can find our blog-post at [Principled Teacher Selection for Knowledge Distillation](https://unprovenalgos.github.io/GRACE).


## Quick Links

- [Repository Structure](#repository-structure)
- [Overview](#overview)
- [Experiments](#experiments)
  - [Prepare Conda Environment](#prepare-conda-environment)
  - [Generate Responses from Teacher](#generate-responses-from-teacher)
  - [Train Student on Teacher's Responses](#train-student-on-teachers-responses)
  - [Evaluate Performance of Trained Model](#evaluate-performance-of-trained-model)
  - [Compute GRACE on the Student's Gradients](#compute-grace-on-the-students-gradients)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## Repository Structure

- `scripts/`: shell scripts for generation, training, evaluation, and GRACE computation
- `src/`: core Python code
- `data/`: datasets
- `models/`: downloaded teacher/student checkpoints
- `output/`: generated responses, trained checkpoints, and evaluation results

## Overview

Training students on teacher-generated responses can significantly improve math reasoning in small models, but selecting the right teacher remains an open question. We introduce GRACE, a cost-efficient gradient-based score that ranks teachers without training the student. On MATH and GSM8K, GRACE achieves up to 86% correlation with the final student performance. When used for teacher selection, the selected teacher enables students to reach within 0.3% of the best achievable performance, outperforming intuitive baselines such as teacher accuracy and student perplexity.

## Experiments

Here, we describe the step-by-step process for generating teacher responses, training the student, evaluating the trained model, and computing GRACE.

### Prepare Conda Environment

Prepare a conda environment using the following command
```Shell
conda env create -f grace.yml
conda activate grace
```
Create folders `models/`, `grace_data/` and `output/` to store model checkpoints and results.
```Shell
mkdir -p models
mkdir -p output
mkdir -p grace_data
```


### Generate Responses from Teacher
First download your favorite teacher to the `models/` folder. Here, we will show with [Qwen-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM

save_dir = 'models/Qwen2.5-3B-Instruct'

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f'Saved model to {save_dir}')
"
```

Now, launch generate.sh. You can modify the temperature of generation from the teacher and the number of responses per question by properly setting TEMPERATURE and RESPONSES. This will store the generated responses at the path specified by `OUTPUT_PATH`, which you can modify in `generate.sh`.


```Shell
sh scripts/generate.sh
```


### Train Student on Teacher's Responses

Now download your favorite student model to the `models/` folder. Here, we will use [LLaMA-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B).

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM

save_dir = 'models/Llama-3.2-1B'

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print(f'Saved model to {save_dir}')
"
```

Now, launch `train.sh`. This script takes the generated responses from the previous step, tokenizes them using the student's tokenizer, and trains the student with the hyperparameters specified in `train.sh`. The final checkpoint will be stored in `output/`, and you can modify the path using `OUTPUT_DIR`.

```Shell
sh scripts/train.sh
```

### Evaluate Performance of Trained Model

Launch eval.sh to evaluate the performance of the trained model. 
```Shell
sh scripts/eval.sh
```

### Compute GRACE on the Student's Gradients

Run `grace.sh` to compute GRACE. The main parameters you may want to adjust are:

- `SUBSAMPLE_QUESTIONS`: number of questions used to compute GRACE (default: `512`)
- `SUBSAMPLE_RESPONSES`: number of responses considered per question (default: `4`)
- `PROJ_DIM`: dimension of the low-dimensional gradient projection (default: `512`)

```Shell
sh scripts/grace.sh
```


## Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Abhishek (ap34 'at' princeton 'dot' edu). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can give more effective help!

## Citation

Please cite our paper if you find our paper or this repo helpful:
```bibtex
@inproceedings{
panigrahi2026in,
title={In Good {GRACES}: Principled Teacher Selection for Knowledge Distillation},
author={Abhishek Panigrahi and Bingbin Liu and Sadhika Malladi and Sham M. Kakade and Surbhi Goel},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=m276fke38H}
}
```