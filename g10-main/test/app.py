# Generative AI was used for creating custom CSS and HTML stylings

import streamlit as st
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Vision Model Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling for navigation and styling
st.markdown("""
<style>
    /* Dark theme */
    .stApp { background-color: #0E1117; }
    
    /* Green accents */
    h1, h2, h3, .stMarkdown { color: #00ff00 !important; }
    .css-1aquho2 { color: #00ff00 !important; }
    
    /* Navigation buttons */
    .nav-button {
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border: 1px solid #00ff00;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        background-color: #00ff0022;
        transform: translateX(5px);
    }
    .nav-button.active {
        background-color: #00ff0044;
        border-left: 4px solid #00ff00;
    }
    
    /* Hide default radio buttons */
    .st-eb [role=radiogroup] {
        display: none;
    }
    
    /* Content containers */
    .content-block {
        padding: 1.5rem;
        border: 1px solid #00ff0033;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    
    /* Copyright */
    .copyright {
        position: fixed;
        bottom: 10px;
        right: 10px;
        color: #00ff00 !important;
        font-size: 0.7em;
        opacity: 0.6;
    }
</style>
""", unsafe_allow_html=True)

# Copyright notice
st.markdown(
    '<div class="copyright">University of Technology, Nuremberg Project</div>',
    unsafe_allow_html=True
)

# Sidebar Navigation
with st.sidebar:
    st.title("🔍 Model Navigator")
    page = st.radio(
        "Select Page",
        ["Project Title", "CNN Analysis", "ViT Analysis", "Model Comparison"],
        format_func=lambda x: f"📌 {x}",
        label_visibility="collapsed"
    )

# Function to render the "Project Title" page
def render_project_title():
    st.title(" Convolution and Attention: A Study on Computer Vision")
    st.markdown("### An Analysis of CNN and Vision Transformer (ViT) Models for Object Detection and Segmentation")
    st.markdown("---")
    st.markdown("""
    Welcome! This project explores the performance of **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** for object detection and segmentation tasks. 
    Use the navigation sidebar to explore the analysis and results.
    """)

# Main Content Area
def render_page():
    if page == "Project Title":
        render_project_title()
    elif page == "ViT Analysis":
        task_type = st.selectbox(
            "🔧 Select Task Type",
            ["Object Detection", "Object Segmentation"],
            key=f"task_{page}"
        )
        
        # Display ViT Model Architecture Description
        st.markdown("""
        ### DeTR (ViT) Model Architecture
        """)
        
        # Placeholder for ViT Architecture Image
        vit_architecture_image_path = "/var/lit2425/pib/g10/DeTRArchitecture.png"  # Replace this with the actual path to your image
        st.image(vit_architecture_image_path, caption="Vision Transformer (ViT) Architecture", use_container_width=True)

        if task_type == "Object Detection":
            # Define main images and associated images for ViT
            vit_main_images = [
                "/var/lit2425/pib/g10/test/AppImages/result/ground_truth/gt_62.png",
                "/var/lit2425/pib/g10/test/AppImages/result/ground_truth/gt_1132.png",
                "/var/lit2425/pib/g10/test/AppImages/result/ground_truth/gt_573.png",
                "/var/lit2425/pib/g10/test/AppImages/result/ground_truth/gt_495.png"
            ]
            vit_associated_images = {
                "image1": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/ground_truth/gt_bbox_62.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/result/dert/62_pretrained.jpg",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/result/dert/62_finetuned.jpg"
                },
                "image2": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/ground_truth/gt_bbox_1132.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/result/dert/1132_pretrained.jpg",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/result/dert/1132_finetuned.jpg"
                },
                "image3": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/ground_truth/gt_bbox_573.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/result/dert/573_pretrained.jpg",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/result/dert/573_finetuned.jpg"
                },
                "image4": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/ground_truth/gt_bbox_495.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/result/dert/495_pretrained.jpg",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/result/dert/495_finetuned.jpg"
                }
            }
            # Call the reusable function for ViT
            display_image_gallery(vit_main_images, vit_associated_images, "selected_image_vit")
            
    elif page == "CNN Analysis":
        task_type = st.selectbox(
            "🔧 Select Task Type",
            ["Object Detection", "Object Segmentation"],
            key=f"task_{page}"
        )
        if task_type == "Object Detection":
            st.markdown("""
            ### Faster RCNN Model Architecture
            """)
            FRCNN_architecture_image_path = "/var/lit2425/pib/g10/FRCNNarch.png"  
            img = Image.open(FRCNN_architecture_image_path)
            img.thumbnail((500, 400))  # Resize the image to a smaller size
            # Create a centered layout using columns
            col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the ratios to control centering
            with col2:
                st.image(img, caption="Faster RCNN Architecture", use_container_width=True)

            FRCNN_main_images = [
                "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/FRCNN_gt1.png",
                "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/FRCNN_gt2.png",
                "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/FRCNN_gt3.png",
                "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/FRCNN_gt4.png"
            ]
            FRCNN_associated_images = {
                "image1": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/FRCNN_gt00.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/pretrained_model_images/image_0.jpg",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/fine-tuned_images/FT1.png"
                },
                "image2": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/FRCNN_gt11.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/pretrained_model_images/image_1.jpg",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/fine-tuned_images/FT2.png"
                },
                "image3": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/FRCNN_gt22.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/pretrained_model_images/image_2.jpg",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/fine-tuned_images/FT3.png"
                },
                "image4": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/FRCNN_gt44.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/pretrained_model_images/image_3.jpg",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/fine-tuned_images/FT4.png"
                }
            }
            # Call the reusable function for ViT
            display_image_gallery(FRCNN_main_images, FRCNN_associated_images, "selected_image_vit")
            
        elif task_type == "Object Segmentation":
            st.markdown("""
            ### Mask RCNN Model Architecture
            """)
            MaskRCNN_architecture_image_path = "/var/lit2425/pib/g10/RCNNarch.png"  
            st.image(MaskRCNN_architecture_image_path, caption="Mask RCNN Architecture", use_container_width=True)

            # Define main images and associated images for ViT
            MRCNN_main_images = [
                "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/image_41888.jpg",
                "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/image_143931.jpg",
                "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/image_296649.jpg",
                "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/image_397133.jpg"
            ]
            MRCNN_associated_images = {
                "image1": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/colored_mask_41888.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/pretrainedoutput4.png",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/output2.png"
                },
                "image2": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/colored_mask_143931.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/pretrainedoutput3.png",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/output4.png"
                },
                "image3": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/colored_mask_296649.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/pretrainedoutput2.png",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/output3.png"
                },
                "image4": {
                    "ground_truth": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/colored_mask_397133.png",
                    "pre_trained": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/pretrainedoutput1.png",
                    "fine_tuned": "/var/lit2425/pib/g10/test/AppImages/result/cocoGT/output1.png"
                }
            }
            # Call the reusable function for ViT
            display_image_gallery(MRCNN_main_images, MRCNN_associated_images, "selected_image")

    elif page == "Model Comparison":
        with st.container():
            st.subheader("📌 Head-to-Head Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Precision Delta",  "ViT leads")
            with col2:
                st.metric("Speed Factor", "18x", "CNN faster")
            with col3:
                st.metric("Memory Usage",  "CNN Better")
            
        # Add markdown to show training times in seconds
        st.markdown("### Inference Time")
        st.markdown(f"- **CNN Inference Time/Image:** 0.0297 seconds")
        st.markdown(f"- **ViT Training Time/Image:** 0.5581 seconds")

# Function to display image gallery
def display_image_gallery(main_images, associated_images, session_state_key):
    # Initialize session state
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = None

    # Display main images with consistent dimensions
    st.markdown("### Select an Image")
    cols = st.columns(4)
    
    for i, img_path in enumerate(main_images):
        with cols[i % 4]:
            # Open image and resize for consistent display
            img = Image.open(img_path)
            img.thumbnail((300, 200))  # Maintain aspect ratio with max dimensions
            
            # Display resized image
            st.image(img, use_container_width=True, caption=f"Image {i+1}")
            
            # Store original path in session state when selected
            if st.button(f"Select Image {i+1}", key=f"select_{i}"):
                st.session_state[session_state_key] = img_path

    # Display selected image and associated results in original size
    if st.session_state[session_state_key]:
        st.markdown("### Selected Image and Associated Results")
        selected_idx = main_images.index(st.session_state[session_state_key])
        selected_key = f"image{selected_idx + 1}"
        
        # Get original sized images
        result_cols = st.columns(4)
        result_cols[0].image(main_images[selected_idx], caption="Main Image")
        result_cols[1].image(associated_images[selected_key]["ground_truth"], caption="Ground Truth")
        result_cols[2].image(associated_images[selected_key]["pre_trained"], caption="Pre-Trained Model")
        result_cols[3].image(associated_images[selected_key]["fine_tuned"], caption="Fine-Tuned Model")

# Render the selected page
render_page()