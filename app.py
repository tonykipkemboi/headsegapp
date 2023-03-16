import streamlit as st
import cv2
import numpy as np
from PIL import Image

import head_segmentation.segmentation_pipeline as seg_pipeline
import head_segmentation.visualization as vis

# Load the pre-trained model
segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline()
visualizer = vis.VisualizationModule()

st.title('Head Segmentation App')

uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
#     st.image(image, caption='Uploaded image', use_column_width=True)

    # Predict the segmentation mask
    segmentation_map = segmentation_pipeline.predict(image)

    # Visualize the segmentation mask
    figure, _ = visualizer.visualize_prediction(image, segmentation_map)
    st.pyplot(figure)

    # Isolate the head using the segmentation mask
    mask = (segmentation_map > 0).astype(np.uint8)
    isolated_head = cv2.bitwise_and(image, image, mask=mask)
    st.image(isolated_head, caption='Isolated head', use_column_width=True)
