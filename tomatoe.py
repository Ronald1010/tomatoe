import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
import tempfile
import os

# Define the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="VsxreoZsgrCDLK4xweXv"
)

MODEL_ID = "tomato-leaf-disease-rxcft/3"

# Dictionary mapping diseases to advice messages
disease_advice = {
    "Late Blight": "Remove affected leaves and apply fungicides.",
    "Septoria": "Use disease-free seeds and rotate crops.",
    "Leaf Mold": "Ensure good air circulation and use fungicides.",
    "Bacterial Spot": "Use copper-based sprays and remove infected plants.",
    "Early Blight": "Apply fungicides and remove infected leaves.",
    "Leaf Miner": "Use insecticides and remove affected leaves."
    # Add more diseases and advice as needed
}

def infer_image(image):
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        image.save(temp_file, format='JPEG')
        temp_file_path = temp_file.name
    
    # Use the inference client to send the image
    result = CLIENT.infer(temp_file_path, model_id=MODEL_ID)
    return result

# Set page title and favicon
st.set_page_config(page_title="Tomato Leaf Disease Detection", page_icon="🍅")

# Define app title and subtitle with emojis
st.title("🍃 Tomato Leaf Disease Detection App")
st.write(
    "This app detects tomato leaf diseases using a pre-trained model. Upload an image to get started! 📷"
)

# Add space for better layout
st.write("")

# Add file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.subheader("Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Displaying preloader while detecting diseases
    with st.spinner("Detecting diseases..."):
        result = infer_image(image)

    # Display results after preloader
    if result and "predictions" in result:
        # Draw bounding boxes on the annotated image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        for prediction in result["predictions"]:
            if all(key in prediction for key in ["x", "y", "width", "height", "class"]):
                x, y, width, height = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
                class_label = prediction["class"]
                # Adjust coordinates to center the bounding box
                x1, y1 = x - width / 2, y - height / 2
                x2, y2 = x + width / 2, y + height / 2
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                # Calculate text size and position
                font_size = int(height * 0.1)  # Set font size to 10% of the bounding box height
                try:
                    font = ImageFont.truetype("arial.ttf", size=font_size)
                except OSError:
                    font = ImageFont.load_default()
                text = f"DISEASE: {class_label.upper()}"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
                text_x1, text_y1 = x1, y1 - text_size[1] - 5  # Position above the bounding box
                text_x2, text_y2 = text_x1 + text_size[0] + 10, text_y1 + text_size[1] + 5
                # Draw background rectangle for text
                draw.rectangle([text_x1, text_y1, text_x2, text_y2], fill="black")
                # Draw text on top of the background rectangle
                draw.text((text_x1 + 5, text_y1 + 2), text, fill="white", font=font)
                
        # Display annotated image
        st.subheader("🖼️ Annotated Image with Bounding Boxes")
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)
        
        # Display number of diseases detected
        num_diseases = len(result["predictions"])
        st.subheader(f"Number of Diseases Detected: {num_diseases}")
        
        # Display bounding box results with larger text and background color
        st.markdown(
            '<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">'
            '<h2 style="color: black;">🔍 Bounding Box Results</h2>',
            unsafe_allow_html=True
        )
        for idx, prediction in enumerate(result["predictions"]):
            disease_class = prediction["class"]
            advice = disease_advice.get(disease_class, "No specific advice available.")
            st.markdown(
                f'<div style="background-color: #ffcccb; padding: 10px; border-radius: 5px; font-size: 20px; color: black;">'
                f'DISEASE: {disease_class.upper()}</div>'
                f'<div style="background-color: #e0e0e0; padding: 10px; border-radius: 5px; font-size: 16px; color: black;">'
                f'ADVICE: {advice}</div>',
                unsafe_allow_html=True
            )
        
    else:
        st.markdown('<p style="color: green; font-weight: bold;">No diseases detected. 😔</p>', unsafe_allow_html=True)
