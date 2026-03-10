# Installation

## 1. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Mac/Linux
venv\Scripts\activate         # On Windows
```

---

## 2. Install PyTorch with CUDA Support

Install PyTorch compatible with **CUDA 11.8**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 3. Install Remaining Dependencies

```bash
pip install numpy pandas scikit-learn tqdm tensorboard matplotlib
```

---

# Dataset Preparation

The following script will:

* Download the **CIFAR-100 dataset**
* Partition the dataset across multiple federated clients
* Generate a configuration file (`cfg.json`) describing the dataset splits

Run the following command:

```bash
python data/main.py \
  --dataset_name cifar100 \
  --n_tasks 20 \
  --save_dir data/cifar100_fed \
  --seed 42 \
  --incongruent_split \
```

## Output

This command will create the directory:

```
data/cifar100_fed/
```


Each client directory contains its own local dataset split.

---

# Running the Benchmark

Once the dataset is prepared, run the federated learning benchmark:

```bash
python benchmark.py \
  --experiment cifar100 \
  --cfg_file_path data/cifar100_fed/cfg.json \
  --device cuda \
  --n_rounds 15 \
  --local_lr 0.01 \
  --results_file logs/benchmark/results_cifar100.csv
```

## Parameter Explanation

| Parameter         | Description                              |
| ----------------- | ---------------------------------------- |
| `--experiment`    | Name of the experiment configuration     |
| `--cfg_file_path` | Path to dataset configuration file       |
| `--device`        | Compute device (`cuda` or `cpu`)         |
| `--n_rounds`      | Number of federated communication rounds |
| `--local_lr`      | Local learning rate used by clients      |
| `--results_file`  | Path to store experiment results         |

---

# Results

The benchmark results will be stored in:

```
logs/benchmark/results_cifar100.csv
```




