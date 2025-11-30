<h1 align='center'>Permutation Equivariant Graph-Neural Controlled Differential Equations <br></h1>

Official implementation of the paper *“Permutation Equivariance in Neural Controlled Differential Equations for Temporal Graph Representation Learning.”*

---

## 💡 Introduction

Building on [Graph Neural Controlled Differential Equations](https://arxiv.org/abs/2009.08295), this repository introduces Permutation Equivariant Graph Neural CDEs, which project Graph Neural CDEs onto permutation equivariant function spaces. This significantly reduces the model’s number of parameters, resulting in more efficient training and improved generalisation, without compromising representational power.

Graph Neural CDEs ([Qin et al. (2023)](https://arxiv.org/abs/2302.11354)) observe dynamic graph topologies as a control path of adjacency matrices $A\colon [0, T] \to \mathbb{R}^{n \times n}$, parameterise a vector field $f_\theta$, and obtain a continuous-time latent state $Z_t$ as the solution to the CDE

$$
Z_t = Z_0 + \int_0^t f_{\theta}(Z_s, A_s) \mathrm{d}A_s.
$$

Due to practical considerations, this is implemented/approximated as

$$
Z_t = Z_0 + \int_0^t Z_s^{(L)} \mathrm{d}s,
$$

where for $l = 1, \dots, L$ we set $Z_s^{(l)} = \sigma\bigl(A_s \, Z_s^{(l-1)} \, W^{(l-1)}\bigr)$ with intitial condition $Z_s^{(0)} = Z_s$ and where adjacency matrix and its time‐derivative are fused via

$$
\widetilde{A}_s =
W^{(F)} \begin{bmatrix}
    A_s \\
    \dfrac{\mathrm{d}A_s}{\mathrm{d}s}
\end{bmatrix},
$$

with $W^{(F)} \in \mathbb{R}^{n \times 2n}$ a learnable fusion matrix and $Z_s^{(0)} = Z_s$. This is inherently not permutation equivariant, and hence we use the maximally expressive basis of permutation equivariant linear functions $\mathbb{R}^{n \times n} \rightarrow \mathbb{R}^{n \times n}$ as characterised by [Maron et al. (2018)](https://arxiv.org/abs/1812.09902):

<details>
<summary>Permutation equivariant basis functions</summary>

1. **Identity and transpose:**  
   $L(\mathbf{A}) = \mathbf{A}, \quad L(\mathbf{A}) = \mathbf{A}^T$  
2. **Eliminate non-diagonal:**  
   $L(\mathbf{A}) = \mathrm{diag}(\mathrm{diag}(\mathbf{A}))$  
3. **Sum of rows (on rows/columns/diagonal):**  
   $L(\mathbf{A}) = \mathbf{1}\mathbf{1}^T\mathbf{A}, \quad L(\mathbf{A}) = \mathbf{1}(\mathbf{A}\mathbf{1})^T, \quad L(\mathbf{A}) = \mathrm{diag}(\mathbf{A}\mathbf{1})$  
4. **Sum of columns (on rows/columns/diagonal):**  
   $L(\mathbf{A}) = \mathbf{A}^T\mathbf{1}\mathbf{1}^T, \quad L(\mathbf{A}) = \mathbf{1}(\mathbf{A}^T\mathbf{1})^T, \quad L(\mathbf{A}) = \mathrm{diag}(\mathbf{A}^T\mathbf{1})$  
5. **Sum of all entries (all entries/diagonal):**  
   $L(\mathbf{A}) = \mathbf{1}^T\mathbf{A}\mathbf{1}\,\mathbf{1}\mathbf{1}^T, \quad L(\mathbf{A}) = \mathbf{1}^T\mathbf{A}\mathbf{1}\,\mathrm{diag}(\mathbf{1})$  
6. **Sum of diagonal entries (all entries/diagonal):**  
   $L(\mathbf{A}) = \mathbf{1}^T\mathrm{diag}(\mathbf{A})\,\mathbf{1}\mathbf{1}^T, \quad L(\mathbf{A}) = \mathbf{1}^T\mathrm{diag}(\mathbf{A})\,\mathrm{diag}(\mathbf{1})$  
7. **Diagonal elements (on rows/columns):**  
   $L(\mathbf{A}) = \mathrm{diag}(\mathbf{A})\mathbf{1}^T, \quad L(\mathbf{A}) = \mathbf{1}\,\mathrm{diag}(\mathbf{A})^T$  

</details>

We define the permutation equivariant fusion as

$$
\bar{A}_s = L_1(A_s) + L_2\Bigl(\tfrac{\mathrm{d}A_s}{\mathrm{d}s}\Bigr),
$$

where $L_1, L_2$ are linear combinations of the basis terms above. The resulting **Permutation Equivariant Graph Neural CDE** is

$$
    Z_t = Z_{t_0} + \int_{t_0}^t \sigma (\bar{A}_s Z^{(L)}_s W^{(L)}) \text{d}s \quad \text{for} \quad t \in ( t_0, t_N ].
$$

---

## 📄 Overview

This repository provides code for training and evaluating Permutation Equivariant Graph Neural Controlled Differential Equations (GNCDEs) on a variety of temporal graph tasks, including:

* Dynamical systems benchmarks
* Pytorch Geometric Temporal datasets
* Temporal Graph Benchmark (TGB) tasks
* Oversampling experiments

---

## 🚀 Getting started

### 1. Clone the repository

```bash
git clone https://github.com/berndtt/perm_equiv_gn_cdes.git
cd perm_equiv_gn_cdes
```

### 2. Create the Conda environment

We recommend Python 3.11. All dependencies are listed in `environment.yaml`.

```bash
conda env create --name perm_equiv_gncdes --file environment.yaml
conda activate perm_equiv_gncdes
```

**Key dependencies:**

* `Jax` & `Diffrax`
* `Equinox`, `Lineax`, `Optax`
* `PyTorch` & `PyTorch Geometric`
* `NumPy`
* `Exca`

---

## 🏃 Running Experiments

### Training the Permutation Equivariant GNCDE

To rerun any of the experiments, run the corresponding script.

```bash
# Dynamical systems benchmark
python src/run/dyn/single_run.py

# Pytorch Geometric Temporal Datasets
python src/run/pgt/single_run.py

# Temporal Graph Benchmark
python src/run/tgb/single_run.py

# Oversampling Experiments
python src/run/dyn/single_run_oversampling.py
```

### Customising Models

To swap in a different model or configuration, edit the YAML config loader in your chosen script:

```python
with open("path/to/config/file.yaml", "r") as file:
    config = yaml.safe_load(file)
```

You can add or modify fields in the config file under the `configs/` directory.

---

## 📊 Dataset

We organize our datasets into three categories—synthetic benchmarks, PyTorch Geometric Temporal (PGT), and Temporal Graph Benchmark (TGB)—and cache them for efficient reuse:

1. **Synthetic Benchmarks**  
   - **Heat Diffusion**, **Gene Regulation**, and **SIR** models  
   - Generated on first run and stored in `.cache/`

2. **PyTorch Geometric Temporal (PGT)**  
   - Pre-downloaded and included under `.datasets/`  
   - Preprocessed and cached in `.cache/`

3. **Temporal Graph Benchmark (TGB)**  
   - Downloaded on demand when first referenced  
   - Preprocessed and cached in `.cache/`

All datasets undergo a one-time preprocessing step, after which the processed files are saved to disk to accelerate subsequent training and evaluation.```

---

## 🔧 Configuration & Logging

We use [Weights & Biases](https://wandb.ai/) for experiment tracking. Ensure you have a local W\&B configuration (`wandb login`).

In your config YAML, set:

```yaml
wandb:
  project: GNCDEs
```

Logs, metrics, and model checkpoints will be automatically synced.

---

## 📚 Acknowledgments

* Our implementation builds upon the codebase of *Learning Dynamic Graph Embeddings with Neural Controlled Differential Equations* by Qin et al.: [https://arxiv.org/abs/2302.11354](https://arxiv.org/abs/2302.11354)
