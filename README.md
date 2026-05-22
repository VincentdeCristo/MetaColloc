# MetaColloc: Optimization-Free PDE Solving via Meta-Learned Basis Functions

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2605.12368-red?logo=arxiv)](https://arxiv.org/abs/2605.12368)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Paper](https://img.shields.io/badge/Hugging%20Face-paper-yellow?logo=huggingface)](https://huggingface.co/papers/2605.12368)

**[Paper](https://arxiv.org/abs/2605.12368)** | **[Hugging Face](https://huggingface.co/papers/2605.12368)**

</div>

This repository contains the official implementation of MetaColloc, an Optimization-Free PDE Solver via Meta-Learned Basis Functions.

## 💡 What is MetaColloc?

Traditional physical information neural networks (PINNs) usually regard the solution of partial differential equations (PDE) as an instance-specific optimization problem. This means that for every new equation or boundary condition, thousands of steps of time-consuming gradient descent need to be performed from scratch.

MetaColloc changes this. We propose to completely decouple "Basis Discovery" and "PDE Solving":

1. Offline Meta-Training: We offline train a two-branch neural network on multi-scale Gaussian random fields (GRF) to build a highly expressive universal neural basis dictionary.
2. Online Test-Time Solving: While testing, network parameters are completely frozen. Solving the new PDE becomes a pure closed-form linear algebra operation (for linear PDEs) or a fast Newton-Raphson iteration (for nonlinear PDEs).

## ⚙️ How it Works

<img width="1298" height="726" alt="image" src="https://github.com/user-attachments/assets/fc6f3d56-f372-4cfd-8e93-37c861fa13ff" />

1. Dual-Branch Architecture: Our network structure cleverly combines the "low-frequency raw coordinate branch" that processes smooth macroscopic signals and the "multi-scale Fourier Features (high-frequency branch)" that captures violent oscillations.
2. Collocation Matrix Assembly: When faced with a new domain, just randomly scatter points (Collocation points) and use forward-mode AutoDiff to extract $\Phi_x, \Phi_{xx}, ...$, and you can instantly assemble the linear equation system $A w = b$ and solve for the coefficient $w$.

## 📄 License

This project is licensed under the [Apache-2.0 License](https://github.com/VincentdeCristo/MetaColloc/blob/main/LICENSE).

## 📰 News

- **[5/22/2026]** Code and experiments released
- **[5/12/2026]** Preprint available on [arXiv](https://arxiv.org/abs/2605.12368)

## 🚀 Quick Start

### 🧪 Configure environment

```bash
mamba create -n metacolloc python=3.12.12
mamba activate metacolloc
mamba install --file requirements.txt
```

## 📧 Contact

**Authors:**

- Zichuan Yang ([2153747@tongji.edu.cn](mailto:2153747@tongji.edu.cn))

**Questions?** Open an [issue](https://github.com/VincentdeCristo/MetaColloc/issues) or email us!

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{yang2026metacollocoptimizationfreepdesolving,
      title={MetaColloc: Optimization-Free PDE Solving via Meta-Learned Basis Functions}, 
      author={Zichuan Yang},
      year={2026},
      eprint={2605.12368},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.12368}, 
}
```
