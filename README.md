## In Good GRACES: Principled Teacher Selection for Knowledge Distillation

This repository contains the code for our paper [In Good GRACES: Principled Teacher Selection for Knowledge Distillation](https://openreview.net/forum?id=m276fke38H), accepted at [ICLR 2026](https://iclr.cc/). You can find our blog-post at [Principled Teacher Selection for Knowledge Distillation](https://unprovenalgos.github.io/GRACE).


## Quick Links

- [Overview](#overview)
- [Experiments](#experiments)
  - [Prepare Conda Environment](#prepare-conda-environment)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## Overview

Training students on teacher-generated responses can significantly improve math reasoning in small models, but selecting the right teacher remains an open question. We introduce GRACE, a cost-efficient gradient-based score that ranks teachers without training the student. On MATH and GSM8K, GRACE achieves up to 86% correlation with the final student performance. When used for teacher selection, the selected teacher enables students to reach within 0.3% of the best achievable performance, outperforming intuitive baselines such as teacher accuracy and student perplexity.




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