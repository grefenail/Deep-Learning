

```markdown
# G10 — Convolution vs Attention: A Study on Computer Vision

## 📌 Overview
This project compares **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** across different computer vision tasks, including **object detection** and **segmentation**.

We evaluate **pretrained models** on the COCO 2017 dataset and then fine-tune both CNN and ViT architectures to measure improvements in **mean Average Precision (mAP)** and **Recall**.

The project includes:
- Jupyter notebooks for **loading datasets**, **evaluating models**, and **fine-tuning pipelines**.
- A **Streamlit dashboard** for visualizing results from fine-tuned models.

---

## 🧠 Models Used
### **CNN-based**
- **Faster R-CNN (ResNet-50 backbone)**  
  - Object detection.
  - Combines a Region Proposal Network (RPN) with an RoI head.
- **Mask R-CNN (ResNet-50 backbone)**  
  - Object segmentation.
  - Extends Faster R-CNN with a RoI mask prediction head.

### **Transformer-based**
- **DETR (DEtection TRansformer)**  
  - Combines a ResNet-50 backbone with a transformer encoder–decoder.
  - Uses attention on features extracted from the backbone.


## 📂 Repository Structure

g10-main/
│
├── src/                          # Source notebooks and scripts
│   ├── Cnn.ipynb                  # Fine-tuning Faster R-CNN for object detection
│   ├── MaskRCNN.ipynb             # Fine-tuning Mask R-CNN for segmentation
│   ├── evaluate_detr.ipynb        # Evaluate DETR (pre-trained and fine-tuned)
│   ├── finetune_detr.ipynb        # Fine-tune DETR on Fashionpedia
│   ├── pretrained_detr_coco.ipynb # Fine-tune DETR on COCO dataset
│
├── test/                          # Dashboard for results visualization
│   └── app.py
│
├── requirements.txt               # Python dependencies
├── requirements_list.txt          # Alternative dependency list
├── README.md                      # Project documentation



## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/grefenail/Deep-Learning.git
cd Deep-Learning/g10-main
````

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### **Running the Jupyter Notebooks**

1. Open Jupyter Lab or Notebook:

   ```bash
   jupyter lab
   ```
2. Set dataset paths in **Cell 2** of the relevant notebook.
3. Run all cells to train, fine-tune, and evaluate models.

### **Running the Dashboard**

```bash
cd test
streamlit run app.py
```

---

## 📊 Datasets

* **COCO 2017** — Common Objects in Context for object detection & segmentation.
* **Fashionpedia** — Fashion-specific segmentation dataset for DETR fine-tuning.

Make sure to update the dataset and annotation paths in the notebooks before running.

---

## 📈 Results Summary

* **Faster R-CNN**: Strong detection performance with efficient training.
* **Mask R-CNN**: Maintains strong detection while adding high-quality segmentation.
* **DETR**: Competitive results using attention mechanisms, especially after fine-tuning.

Detailed quantitative results are available in the notebooks and dashboard.

---

## 👨‍💻 Authors & Acknowledgment

Developed by:

* **Adam Lo Jen Khai**
* **Akash**
* **Priyam**

Special thanks to **Professor Dr. Florian Walter** for guidance during the **Machine Learning Course (WS24/25)** at **University of Technology, Nuremberg**.

---

## 📜 License

This project is for educational purposes as part of the Machine Learning course at UTN.
If you plan to use or modify it, please credit the authors.

```

---

