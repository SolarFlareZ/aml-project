# Federated Learning with Sparse Task Arithmetic on CIFAR-100

This repository implements a Federated Learning (FedAvg) with sparse fine-tuning and task arithmetic, using a DINO-pretrained Vision Transformer backbone on CIFAR-100.

## Features

- Centralized baseline training
- Federated learning (IID / non-IID)
- Fisher Information–based sparse fine-tuning
- Task arithmetic with scalable task vectors
- Fully reproducible experiments via Hydra configuration



##  Project Structure

```text
.
├── configs/                    # Hydra YAML configurations
│   ├── base.yaml
│   ├── centralized.yaml
│   └── fedavg.yaml
│
├── src/                        # Core implementation
│   ├── datamodule.py           # Centralized & federated data handling
│   ├── model.py                # DINO + linear classifier
│   ├── federated.py            # FedAvg implementation
│   ├── pruner.py               # Fisher-based mask calibration
│   ├── sparse_optimizer.py     # SparseSGDM optimizer
│   ├── train.py                # Centralized training script
│   ├── train_fedavg.py         # Federated training script
│   └── eval_task_arithmetic.py # Task arithmetic evaluation
│
├── notebooks/
│   ├── 00_centralized_baseline.ipynb
│   ├── 01_federated_baseline.ipynb
│   └── 02_task_arithmetic.ipynb
│
├── results/                    # Logs, checkpoints, outputs
├── requirements.txt
└── README.md



##  Environment Setup


### 1. Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt


Note: The DINO backbone is loaded via torch.hub and requires an active internet connection on first run.

Dataset

The project uses CIFAR-100 via torchvision.

The dataset is downloaded automatically

Default location: ./data

No manual download is required.

🧪 Experiments and Reproducibility

All experiments are controlled via Hydra YAML configurations located in configs/.

Configuration hierarchy:

base.yaml → shared settings

centralized.yaml → centralized baseline

fedavg.yaml → federated learning + sparse task arithmetic

1️⃣ Centralized Baseline Training

This experiment trains a centralized classifier on CIFAR-100 and serves as the baseline model.

Run command
python -m src.train

Configuration used

configs/centralized.yaml

Automatically loads base.yaml

Outputs

Best model checkpoint (by validation accuracy)

Training and validation logs

Test accuracy of the best checkpoint

Purpose

Reference model for comparison with federated learning

Source of the base model in task arithmetic

2️⃣ Federated Learning (FedAvg)

This experiment runs Federated Averaging with configurable client participation, local steps, and data sharding.

Run command
python -m src.train_fedavg

Configuration used

configs/fedavg.yaml

Automatically loads base.yaml

Key configurable parameters

num_clients

participation_rate

local_steps

num_rounds

sharding (iid or non_iid)

Outputs

Federated model checkpoints

Per-round validation accuracy

Final test accuracy

JSON file with full experiment history

3️⃣ Sparse Fine-Tuning (Fisher-Based)

Sparse fine-tuning is enabled inside federated learning by setting:

federated:
  use_sparse: true

Process

Ridge initialization of the classifier head

Fisher Information calibration on validation data

Selection of least-sensitive parameters

Sparse updates using SparseSGDM

Relevant config fields
use_sparse: true
sparsity_level: 0.5
num_calibration_rounds: 3
sparse_strategy: least_sensitive

4️⃣ Task Arithmetic Evaluation

This step evaluates linear combinations of the centralized and federated models.

Mathematical form
𝑊
new
=
𝑊
base
+
𝛼
(
𝑊
fed
−
𝑊
base
)
W
new
	​

=W
base
	​

+α(W
fed
	​

−W
base
	​

)
Run command
python -m src.eval_task_arithmetic

What it does

Loads the centralized baseline model

Loads the federated model

Sweeps over multiple values of α

Evaluates each combined model on the test set

Output

Table of test accuracies for different α values

Identification of optimal task arithmetic scaling

📓 Jupyter Notebooks (Optional)

The notebooks/ folder contains interactive notebooks for understanding and debugging:

Notebook	Purpose
00_centralized_baseline.ipynb	Sanity check centralized training
01_federated_baseline.ipynb	Sanity check FedAvg without extensions
02_task_arithmetic.ipynb	Interactive task arithmetic analysis

⚠️ Important:
All reported results should be obtained from the Python scripts, not from notebooks.
Notebooks are provided for exploration and validation only.

🔁 Reproducibility Notes

All randomness is controlled via a global seed

Experiments are fully config-driven

Checkpoints and logs are stored under results/

Hydra ensures consistent experiment instantiation


How to Reproduce the Results

This section describes the exact procedure required to reproduce the experimental results reported in this work. All experiments are fully configuration-driven and can be reproduced using the provided scripts and configuration files.

Environment Setup

All experiments were conducted using Python and PyTorch. To reproduce the results, first create a clean virtual environment and install the required dependencies:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


The DINO backbone is automatically downloaded via torch.hub during the first execution and requires an active internet connection.

Dataset Preparation

The CIFAR-100 dataset is used for all experiments. The dataset is automatically downloaded using torchvision and stored in the directory specified in configs/base.yaml (default: ./data). No manual dataset preparation is required.

Centralized Baseline Training

The centralized baseline model is trained using the configuration defined in configs/centralized.yaml.

To reproduce the centralized results, run:

python -m src.train


This script performs supervised training on the full CIFAR-100 training set, applies validation-based early stopping, and evaluates the best-performing checkpoint on the test set. The resulting model serves as the baseline reference for all subsequent federated and task arithmetic experiments.

Federated Learning with FedAvg

Federated learning experiments are executed using the configuration defined in configs/fedavg.yaml, which extends the shared settings in configs/base.yaml.

To reproduce the federated learning results, run:

python -m src.train_fedavg


This script simulates federated learning with a configurable number of clients, client participation rate, number of local update steps, and total communication rounds. Both IID and non-IID data distributions are supported and can be selected via the configuration file.

Sparse Fine-Tuning via Fisher Information

Sparse fine-tuning is enabled within federated learning by setting use_sparse: true in configs/fedavg.yaml. When enabled, the following steps are automatically executed:

Ridge-based initialization of the classifier head

Fisher Information–based calibration over multiple rounds

Selection of least-sensitive parameters according to the specified sparsity level

Sparse local updates using a masked SGD optimizer

No additional commands are required beyond running the federated training script.

Task Arithmetic Evaluation

To evaluate the effect of task arithmetic, the centralized baseline model and the final federated model are linearly combined using a scaling factor α.

To reproduce the task arithmetic results, run:

python -m src.eval_task_arithmetic


This script loads the trained centralized and federated models, applies task arithmetic for a range of α values, and evaluates each combined model on the CIFAR-100 test set. The reported accuracies correspond to these evaluations.

Notes on Reproducibility

All experiments use a fixed random seed defined in configs/base.yaml

Hyperparameters are entirely controlled via Hydra configuration files

All checkpoints, logs, and result files are stored under the results/ directory

Jupyter notebooks are provided only for interactive inspection and debugging; all reported results are obtained from the Python scripts
