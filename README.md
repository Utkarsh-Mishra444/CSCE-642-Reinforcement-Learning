# VGAP: Vision-Gated Action Planning for Web Agents

A DPO-trained visual grounding model that optimizes screenshot cropping for web automation agents, improving action accuracy by focusing on relevant UI regions.

## Overview

VGAP (Vision-Gated Action Planning) trains a lightweight language model (Qwen2-0.5B) using Direct Preference Optimization (DPO) to predict optimal crop regions for web screenshots. Instead of processing full-page screenshots, VGAP identifies the most relevant area containing the target UI element with appropriate context.

### Key Features

- **Smart Cropping**: Learns to predict optimal `CROP(x1,y1,x2,y2)` regions around target elements
- **DPO Training**: Uses preference pairs (good crops vs. bad crops) for training
- **Lightweight**: Based on Qwen2-0.5B (~500M parameters)
- **Colab-Ready**: Designed for training on Google Colab with L4/A100 GPUs

## Repository Structure

```
├── train/                          # Training scripts
│   ├── create_preferences.py       # Generate DPO preference pairs from Mind2Web
│   ├── download_mind2web.py        # Dataset download utility
│   ├── train_vgap.py              # Main DPO training script
│   └── train_vgap_plus_eval.py    # Training with evaluation
├── Agent/                          # Web agent integration
│   ├── SeeAct/                    # SeeAct framework (modified)
│   └── test.py                    # Example agent usage
└── README.md
```

## Quick Start (Google Colab)

### Prerequisites

- Google Colab with L4 or A100 GPU
- Google Drive for persistent storage

### Cell 1: Setup

```python
# Install dependencies
!pip install -q transformers peft trl bitsandbytes datasets accelerate tqdm

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone or upload this repo to Drive, then verify
!ls /content/drive/MyDrive/VGAP/
```

### Cell 2: Generate Training Data (~20-30 min)

```python
# Generate 2000 training examples → ~6000 preference pairs
!python /content/drive/MyDrive/VGAP/train/create_preferences.py \
    --split train \
    --num_samples 2000 \
    --output /content/drive/MyDrive/VGAP/data/dpo_train_2k.json
```

### Cell 3: Generate Test Data (~3-5 min)

```python
# Generate 200 test examples → ~600 preference pairs
!python /content/drive/MyDrive/VGAP/train/create_preferences.py \
    --split test_task \
    --num_samples 200 \
    --output /content/drive/MyDrive/VGAP/data/dpo_test_200.json
```

### Cell 4: Train Model (~2-3 hours on L4)

```python
# Train with experiment versioning
!python /content/drive/MyDrive/VGAP/train/train_vgap.py \
    --data_path /content/drive/MyDrive/VGAP/data/dpo_train_2k.json \
    --experiments_dir /content/drive/MyDrive/VGAP/experiments \
    --experiment_name vgap_2k_v1 \
    --epochs 3 \
    --batch_size 4
```

### Cell 5: Evaluate on Test Set

```python
# Evaluate trained model
!python /content/drive/MyDrive/VGAP/train/train_vgap.py \
    --data_path /content/drive/MyDrive/VGAP/data/dpo_test_200.json \
    --load_model /content/drive/MyDrive/VGAP/experiments/vgap_2k_v1
```

---

## Detailed Documentation

### Data Generation

The `create_preferences.py` script downloads from HuggingFace's [Multimodal-Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) dataset and creates DPO preference pairs.

**Available Splits:**

| Split | Description | Use Case |
|-------|-------------|----------|
| `train` | Training data (~7000 total) | Model training |
| `test_task` | New tasks, same websites | Primary evaluation |
| `test_website` | New websites | Cross-website generalization |
| `test_domain` | New domains | Cross-domain generalization |

**Command Options:**

```bash
python train/create_preferences.py \
    --split train           # Dataset split
    --num_samples 2000      # Number of samples (None = all)
    --output path/to/output.json
```

### Training Configuration

The `train_vgap.py` script handles DPO training with automatic experiment versioning.

**Key Options:**

```bash
python train/train_vgap.py \
    --data_path PATH              # Path to preference pairs JSON
    --experiments_dir PATH        # Base dir for experiments
    --experiment_name NAME        # Custom name (auto-generates if not set)
    --epochs 3                    # Number of training epochs
    --batch_size 4                # Batch size per device
    --lr 5e-5                     # Learning rate
    --use_4bit                    # Use 4-bit quantization (optional)
```

**Output Structure:**

```
experiments/
└── vgap_2k_v1/
    ├── adapter_config.json         # LoRA configuration
    ├── adapter_model.safetensors   # LoRA weights (~70MB)
    ├── logs/                       # Training logs
    └── merged/                     # Full merged model (~1GB)
        ├── config.json
        └── model.safetensors
```

### Evaluation

Load a trained model and evaluate:

```bash
python train/train_vgap.py \
    --data_path /path/to/test_data.json \
    --load_model /path/to/experiments/vgap_2k_v1
```

---

## Time Estimates (L4 GPU)

| Task | Time |
|------|------|
| Data generation (2000 train) | ~20-30 min |
| Data generation (200 test) | ~3-5 min |
| Training (3 epochs, 6000 pairs) | ~2-3 hours |
| Evaluation (200 test) | ~5-10 min |
| **Total** | **~3-4 hours** |

---

## Experiment Versioning

Experiments are automatically versioned to prevent overwrites:

```bash
# Custom name → experiments/vgap_2k_v1/
--experiment_name vgap_2k_v1

# Auto-generated → experiments/vgap_20241207_143022/
# (no --experiment_name flag)

# If name exists, timestamp is appended:
# experiments/vgap_2k_v1_20241207_150000/
```

---

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
--batch_size 2
```

Or enable 4-bit quantization (not recommended - can cause unstable training):
```bash
--use_4bit
```

### Slow Data Generation

HuggingFace streaming can be slow. Data is cached to Drive, so subsequent runs are faster.

### Session Disconnected

All checkpoints are saved to Drive. Resume by:
1. Re-mount Drive
2. Continue from last checkpoint (auto-saved every 50 steps)

---

## Full Scale Training

Once 2000 examples work well, scale to full dataset:

```python
# Generate all training data (~1-2 hours)
!python /content/drive/MyDrive/VGAP/train/create_preferences.py \
    --split train \
    --output /content/drive/MyDrive/VGAP/data/dpo_train_full.json

# Train on full data (~8-10 hours)
!python /content/drive/MyDrive/VGAP/train/train_vgap.py \
    --data_path /content/drive/MyDrive/VGAP/data/dpo_train_full.json \
    --experiments_dir /content/drive/MyDrive/VGAP/experiments \
    --experiment_name vgap_full_v1 \
    --epochs 3 \
    --batch_size 4
```

---

## SeeAct Integration


# VGAP-Integrated SeeAct

This fork of [SeeAct](https://github.com/OSU-NLP-Group/SeeAct) integrates **VGAP (Visual Grounding for Action Planning)** as an intelligent screenshot preprocessing module and adds **universal model support** via LiteLLM.

## 🚀 Key Features

- **VGAP Cropping**: Automatically crop screenshots to task-relevant regions before action prediction
- **Universal Model Support**: Use any vision-language model supported by LiteLLM (100+ models)
- **Local Inference**: Run with locally-hosted models via Ollama (zero API cost, full privacy)
- **Graceful Fallback**: Falls back to full screenshots if VGAP fails

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SeeAct.git
cd SeeAct/seeact_package

# Install dependencies
pip install -e .
pip install litellm pillow requests
```

### For Local VGAP Model (Optional)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Create and load the VGAP model from GGUF
ollama create vgap-v1 -f /path/to/Modelfile
```

Example `Modelfile`:
```
FROM ./vgap-model.gguf
PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER num_predict 200
TEMPLATE """{{ .Prompt }}"""
```

## ⚙️ Configuration

### New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_vgap_cropping` | bool | `False` | Enable VGAP screenshot cropping |
| `vgap_model` | str | `"ollama/vgap-2k-v1:latest"` | Model for VGAP inference |
| `vgap_max_retries` | int | `3` | Max retry attempts for VGAP |
| `model` | str | `"gpt-4o"` | Action model (any LiteLLM-compatible model) |

## 📖 Usage

### Basic Usage (Without VGAP)

```python
import asyncio
from seeact.agent import SeeActAgent

async def main():
    agent = SeeActAgent(
        model="gpt-4o",  # Or any LiteLLM model
    )
    await agent.start()
    await agent.execute(
        "https://www.google.com",
        "Search for 'web automation research'"
    )
    await agent.stop()

asyncio.run(main())
```

### With VGAP Cropping Enabled

```python
import asyncio
from seeact.agent import SeeActAgent

async def main():
    agent = SeeActAgent(
        model="gpt-4o",
        use_vgap_cropping=True,
        vgap_model="ollama/vgap-v1:latest",  # Local VGAP model
        vgap_max_retries=3,
    )
    await agent.start()
    await agent.execute(
        "https://www.expedia.com",
        "Book a flight from New York to Los Angeles"
    )
    await agent.stop()

asyncio.run(main())
```

### Using Different Action Models

```python
# OpenAI
agent = SeeActAgent(model="gpt-4o")
agent = SeeActAgent(model="gpt-4-turbo")
agent = SeeActAgent(model="gpt-4.1")

# Anthropic
agent = SeeActAgent(model="anthropic/claude-sonnet-4-20250514")
agent = SeeActAgent(model="anthropic/claude-3-5-sonnet-20241022")

# Google
agent = SeeActAgent(model="gemini/gemini-1.5-pro")
agent = SeeActAgent(model="gemini/gemini-2.0-flash")

# Local via Ollama
agent = SeeActAgent(model="ollama/llava:latest")
agent = SeeActAgent(model="ollama/llama3.2-vision:latest")

# Azure OpenAI
agent = SeeActAgent(model="azure/gpt-4o")
```

## 🔧 Environment Variables

```bash
# For cloud models
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."

# For debugging LiteLLM requests
export SEEACT_LITELLM_VERBOSE=true
```

## 📁 Output Structure

When VGAP cropping is enabled, the agent creates additional logs:

```
seeact_agent_files/
└── 20251209_123456/
    ├── agent.log              # Main agent log
    ├── screenshots/
    │   ├── screen_1.png       # Full screenshot
    │   ├── screen_1_cropped.png  # VGAP-cropped screenshot
    │   ├── screen_2.png
    │   └── screen_2_cropped.png
    └── vgap_logs/
        ├── vgap_prompt_1.txt  # VGAP input prompt
        └── vgap_prompt_2.txt
```

## 🔍 How VGAP Works

1. **Element Extraction**: Extract interactive elements from DOM with bounding boxes
2. **Prompt Construction**: Format elements for VGAP: `[idx] <tag>description</tag> bbox:(x1,y1,x2,y2)`
3. **VGAP Inference**: Query VGAP model to predict optimal crop region
4. **Postprocessing**: Parse coordinates with multi-strategy fallback
5. **Cropping**: Crop screenshot to predicted region
6. **Action Prediction**: Pass cropped image to action model

## 📊 Supported Models (via LiteLLM)

| Provider | Example Models |
|----------|----------------|
| OpenAI | gpt-4o, gpt-4-turbo, gpt-4.1, gpt-4o-mini |
| Anthropic | claude-sonnet-4-20250514, claude-3-5-sonnet, claude-3-opus |
| Google | gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash |
| Mistral | mistral-large, pixtral-large |
| Ollama | llava, llama3.2-vision, bakllava |
| Azure | azure/gpt-4o, azure/gpt-4-turbo |
| AWS Bedrock | bedrock/anthropic.claude-3 |
| And many more... | See [LiteLLM docs](https://docs.litellm.ai/docs/providers) |


---

## Requirements

```
transformers>=4.35.0
peft>=0.6.0
trl>=0.7.0
bitsandbytes>=0.41.0
datasets>=2.14.0
accelerate
tqdm
torch>=2.0.0
```

---

## Citations

If you use this work, please cite the following:

### This Project

```bibtex
@misc{vgap2024,
  title={VGAP: Vision-Gated Action Planning for Web Agents},
  year={2024}
}
```

### Qwen2 (Base Model)

```bibtex
@article{yang2024qwen2,
  title={Qwen2 Technical Report},
  author={Yang, An and Yang, Baosong and Hui, Binyuan and Zheng, Bo and Yu, Bowen and Zhou, Chang and Li, Chengpeng and Li, Chengyuan and Liu, Dayiheng and Huang, Fei and others},
  journal={arXiv preprint arXiv:2407.10671},
  year={2024}
}
```

### Direct Preference Optimization (DPO)

```bibtex
@inproceedings{rafailov2024direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher D and Ermon, Stefano and Finn, Chelsea},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

### TRL (Transformer Reinforcement Learning)

```bibtex
@misc{vonwerra2022trl,
  title={TRL: Transformer Reinforcement Learning},
  author={von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/huggingface/trl}},
  year={2022}
}
```

### PEFT (Parameter-Efficient Fine-Tuning)

```bibtex
@misc{peft,
  title={PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author={Mangrulkar, Sourab and Gugger, Sylvain and Debut, Lysandre and Belkada, Younes and Paul, Sayak and Bossan, Benjamin},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/huggingface/peft}},
  year={2022}
}
```

### LoRA (Low-Rank Adaptation)

```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

### Mind2Web (Dataset)

```bibtex
@inproceedings{deng2023mindweb,
  title={Mind2Web: Towards a Generalist Agent for the Web},
  author={Deng, Xiang and Gu, Yu and Zheng, Boyuan and Chen, Shijie and Stevens, Samuel and Wang, Boshi and Sun, Huan and Su, Yu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023},
  url={https://openreview.net/forum?id=kiYqbO3wqw}
}
```

### SeeAct (Web Agent Framework)

```bibtex
@inproceedings{zheng2024seeact,
  title={GPT-4V(ision) is a Generalist Web Agent, if Grounded},
  author={Zheng, Boyuan and Gou, Boyu and Kil, Jihyung and Sun, Huan and Su, Yu},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=piecKJ2DlB}
}
```

### Hugging Face Transformers

```bibtex
@inproceedings{wolf2020transformers,
  title={Transformers: State-of-the-Art Natural Language Processing},
  author={Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Pierric and Rault, Tim and Louf, Remi and Funtowicz, Morgan and others},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages={38--45},
  year={2020}
}
```

---

## Acknowledgments

- [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) for the dataset
- [SeeAct](https://github.com/OSU-NLP-Group/SeeAct) for the web agent framework
- [Hugging Face](https://huggingface.co/) for Transformers, PEFT, and TRL libraries
- [Qwen Team](https://github.com/QwenLM/Qwen2) for the Qwen2 model
- GitHub Copilot for assisting with boilerplate and refactoring during code development. 

