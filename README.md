Convolution vs Attention: A study on Computer Vision
Description
In this project, we use CNNs and ViTs across different computer vision tasks like object detection and segmentation. We first evaluate the pretrained models on COCO 2017 dataset and then compare the improvement of fine-tuning CNN and ViT models in terms of mean Average Precision and Recall.

We have developed Jupyter notebooks with complete functionality to load COCO-style datasets, load and evaluate pretrained models, and a complete fine-tuning pipeline for object detection and segmentation tasks. Additionally, we have created a dashboard to visualize results from our fine-tuned models.

Models Used
FasterRCNN : For object detection with CNN we have used Resnet50 based Faster RCNN pretrained model. It combines a Region Proposal Network (RPN) with RoI head for object detection.
MaskRCNN : For object segmentation with CNN we have used Resnet50 based MakRCNN pretrained model. It extends Faster R-CNN by adding a RoI mask prediction head for parallel segmentation.
DETR : For ViT, we have used the DETR model, which combines a ResNet-50 backbone with a transformer encoder-decoder. using attention on features extracted from the ResNet backbone.
Installation
Usage
Install the required dependencies in your virtual environment.
pip install -r requirements.txt
Note: Make sure to update the path to your COCO-style dataset and annotations in Cell 2.
Once the paths are set correctly, you can use the jupyter notebooks. After that, everything should work fine.
To run the visual dashboard,
cd test
streamlit run app.py
Files
All the files stated below are from src folder:

Cnn.ipynb: This notebook focuses on fine-tuning a pretrained Faster R-CNN model using a ResNet50 backbone for object detection tasks on COCO dataset.
MaskRCNN.ipynb: This notebook is dedicated to fine-tuning a pretrained Mask R-CNN model with a ResNet50 backbone for object segmentation tasks COCO dataset.
evaluate_detr.ipynb: This notebook is dedicated to evaluate performance of DETR (pre-trained and fine-tuned) on COCO dataset.
finetune_detr.ipynb: This notebook is dedicated to fine-tuning DETR on Fashionpedia dataset.
pretrained_detr_coco.ipynb: This notebook is dedicated to fine-tuning DETR on COCO datastet.
Download our fine-tuned model from here

Authors and acknowledgment
This project was developed by Adam, Akash and Priyam as part of the Machine Learning Course WS24/25 at University of Technology, Nuremberg. Special thanks to Professor Dr. Florian Walter for his guidance throughout the project and course.
