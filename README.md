# Generative Graph Diffusion Model for Wireless Power and Fronthaul Bandwidth Allocation in Low-Altitude Assisted Cell-Free Networks

**Abstract:** Low-altitude assisted cell-free networks are a promising solution for achieving cooperative aerial-ground coverage through centralized control. However, wireless access and fronthaul links are typically supported within different spectrum bands which complicates aerial-ground radio resource management. Meanwhile, wireless fronthaul bandwidth utilization restricts its transmission capacity to further influence network capacity. This paper investigates joint allocation of wireless power and fronthaul bandwidth to maximize spectrum efficiency in low-altitude assisted cell-free networks. We first exploit a deterministic optimization method to obtain near-optimal solutions as training samples. Then, we represent the considered network as a heterogeneous graph data and propose a generative graph diffusion model to generate high-quality solutions by learning the distribution of obtained training samples. Specifically, the proposed model first forward diffuses heterogeneous graph data by adding Gaussian noise and then conducts reverse diffusion based on our customized graph U-Net model. Simulation results demonstrate that our proposed GUN-Diff model can approach the performance of deterministic optimization methods with high computation efficiency and well generative ability across different network environments and configurations.

This repository contains the code accompanying the paper:

> **"Generative Graph Diffusion Model for Wireless Power and Fronthaul Bandwidth Allocation in Low-Altitude Assisted Cell-Free Networks"**

---

## 💻  Activate Coding Environment

To create a new conda environment, execute the following command:

```bash
conda create --name gun-diff python==3.12.7
```

Activate the created environment with:

```bash
conda activate gun-diff
```

## ⚡ Install Required Packages

The packages can be installed using:

```bash
pip install -r requirements.txt
```

## 🚀Get Dataset for training

Run `get_dataset_multi.py` in forder `get_dataset` to get dataset

## ▶️ Run the Program

Run  `main.py`  to start the program.

## 🔍 Check the Results

1. After the model is well-trained, you can check the model in folder `model` and the relative data in folder `data`.
2. You can run `draw.py` in folder `draw` to verify the results.

---
