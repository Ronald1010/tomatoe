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
    "Bacterial Spot": (
        "1. Remove and destroy infected plant debris to prevent the spread of the bacteria.\n"
        "2. Apply copper-based sprays every 7-10 days to help control the disease.\n"
        "3. Avoid overhead watering to reduce leaf wetness and the potential for bacterial spread.\n"
        "4. Practice crop rotation to reduce the presence of the bacteria in the soil.\n"
        "5. Use disease-free seeds and resistant plant varieties if available."
    ),
    "Early Blight": (
        "1. Apply fungicides such as chlorothalonil or copper-based products at the first sign of disease and continue at regular intervals.\n"
        "2. Remove and destroy infected leaves to reduce the spread of the fungus.\n"
        "3. Mulch around the base of plants to reduce soil splash onto leaves, which can spread the pathogen.\n"
        "4. Practice crop rotation and avoid planting tomatoes or potatoes in the same location each year.\n"
        "5. Ensure good air circulation by spacing plants properly and pruning excess foliage."
    ),
    "Healthy": (
        "1. Continue regular monitoring of plants for any signs of disease or pest infestation.\n"
        "2. Maintain good garden hygiene by removing dead leaves and debris regularly.\n"
        "3. Provide plants with adequate water and nutrients to ensure optimal growth.\n"
        "4. Use organic mulches to help retain soil moisture and reduce weed competition.\n"
        "5. Rotate crops annually to prevent the buildup of soil-borne diseases."
    ),
    "Iron Deficiency": (
        "1. Apply iron chelates to the soil or as a foliar spray to correct iron deficiency.\n"
        "2. Adjust soil pH to between 6.0 and 6.5, as iron is more available to plants in slightly acidic soils.\n"
        "3. Avoid overwatering, as waterlogged soils can inhibit iron uptake.\n"
        "4. Ensure proper fertilization, avoiding excessive phosphorus, which can interfere with iron absorption.\n"
        "5. Plant iron-efficient varieties if available, especially in areas prone to iron deficiency."
    ),
    "Late Blight": (
        "1. Remove and destroy all affected leaves and plants to prevent the spread of the pathogen.\n"
        "2. Apply fungicides such as chlorothalonil, copper-based products, or specific late blight fungicides regularly, especially in wet conditions.\n"
        "3. Avoid overhead watering to reduce leaf wetness.\n"
        "4. Space plants properly to ensure good air circulation and reduce humidity around the plants.\n"
        "5. Practice crop rotation and avoid planting tomatoes and potatoes in the same area each year."
    ),
    "Leaf Mold": (
        "1. Ensure good air circulation by properly spacing plants and pruning excess foliage.\n"
        "2. Apply fungicides such as copper-based products or other fungicides labeled for leaf mold.\n"
        "3. Water plants at the base to avoid wetting the foliage.\n"
        "4. Remove and destroy infected leaves to reduce the source of inoculum.\n"
        "5. Grow resistant varieties if available."
    ),
    "Leaf_Miner": (
        "1. Use insecticides such as spinosad or neem oil to control leaf miner populations.\n"
        "2. Remove and destroy affected leaves to reduce the number of larvae developing into adults.\n"
        "3. Use yellow sticky traps to monitor and control adult leaf miners.\n"
        "4. Encourage natural predators, such as parasitic wasps, which can help control leaf miner populations.\n"
        "5. Practice crop rotation to disrupt the life cycle of the pests."
    ),
    "Mosaic Virus": (
        "1. Remove and destroy infected plants to prevent the spread of the virus.\n"
        "2. Control aphids and other insect vectors that can transmit the virus using insecticidal soaps or oils.\n"
        "3. Avoid working with wet plants to reduce the risk of spreading the virus through contact.\n"
        "4. Disinfect tools and hands after handling infected plants.\n"
        "5. Use virus-free seeds and resistant varieties if available."
    ),
    "Septoria": (
        "1. Use disease-free seeds and transplants to prevent introducing the pathogen.\n"
        "2. Apply fungicides such as chlorothalonil or mancozeb at the first sign of disease and continue at regular intervals.\n"
        "3. Remove and destroy infected leaves to reduce the spread of the fungus.\n"
        "4. Practice crop rotation and avoid planting tomatoes in the same location each year.\n"
        "5. Ensure good air circulation by spacing plants properly and pruning excess foliage."
    ),
    "Spider Mites": (
        "1. Spray plants with water to dislodge spider mites and reduce their numbers.\n"
        "2. Apply miticides or insecticidal soaps to control spider mite populations.\n"
        "3. Introduce natural predators, such as ladybugs or predatory mites, to help manage spider mite infestations.\n"
        "4. Maintain adequate humidity around plants, as spider mites thrive in dry conditions.\n"
        "5. Regularly inspect plants for early signs of spider mite activity and take prompt action."
    ),
    "Yellow Leaf Curl Virus": (
        "1. Remove and destroy infected plants to prevent the spread of the virus.\n"
        "2. Control whiteflies, the primary vector of the virus, using insecticidal soaps, oils, or yellow sticky traps.\n"
        "3. Avoid planting tomatoes near crops that are hosts for whiteflies.\n"
        "4. Use virus-free transplants and resistant varieties if available.\n"
        "5. Practice good garden hygiene by removing weeds and plant debris that can harbor whiteflies and the virus."
    )
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
            '<h2 style="color: black;">🔍 Disease Detection Results</h2>',
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
