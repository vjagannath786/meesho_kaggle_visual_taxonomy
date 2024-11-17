
# Meesho Kaggle Visual Taxonomy

This repository is designed for training and inference on the Meesho Kaggle Visual Taxonomy dataset. Follow the steps below to set up, train, and infer from the model.

---

## **Setup Instructions**

### 1. Download the Data
- Download the dataset and place it in the `data/` directory.

### 2. Install Dependencies
- Install the required Python packages using:
  ```bash
  pip install -r requirements.txt
  ```

---

## **Usage**

### Please note code best runs with A100 GPU's

### **Training**
#### **Single GPU**
Run the following command to train the model on a single GPU:
```bash
python training.py
```

#### **Multi-GPU**
For multi-GPU training, use the `torchrun` command:
```bash
torchrun --nproc_per_node=<gpu_count> training.py
```
Replace `<gpu_count>` with the number of GPUs available on your machine.

---

### **Inference**
#### **Setup**
1. Download the pre-trained merged model from the provided link.
2. Place the model in the `models/` directory.

#### **Run Inference**
Execute the following command to perform inference:
```bash
python inference.py
```

---

## **Folder Structure**
Your project should follow this structure:
```
meesho_kaggle_visual_taxonomy/
│
├── data/                   # Dataset directory
├── adapters/               # adapter directory where adapter is saved
├── models/                 # Pre-trained models directory
├── training.py             # Training script
├── inference.py            # Inference script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## **Notes**
- Ensure that the `data/` and `models/` directories are set up correctly before running the scripts.
- For multi-GPU training, ensure `CUDA_VISIBLE_DEVICES` is correctly configured.