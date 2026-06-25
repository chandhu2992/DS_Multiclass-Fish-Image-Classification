from gettext import install
import streamlit as st
import tensorflow
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np 

# Load the saved model
model = load_model('MobileNet.h5')

# Define the fish categories
fish_classes = ['animal fish', 'animal fish bass','fish sea_food black_sea_sprat','fish sea_food glit_head_bream','fish sea_food hourse_mackerel',
'fish sea_food red_mullet','fish sea_food red_sea_breem','fish sea_food sea_bass','fish sea_food shrimp','fish sea_food striped_red_mullet',
'fish sea_food trout']

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to the input shape expected by the model
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app interface
st.title('Fish Image Classifier')

# Upload image
uploaded_image = st.file_uploader('Upload a fish image', type=['jpg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])  # Adjust the column widths if needed

    # Display the image in the first column
    with col1:
        st.image(image, caption='Uploaded Fish Image', width=300)  # Display a smaller image to reduce size

    # Preprocess the image for model prediction
    preprocessed_image = preprocess_image(image)

    # Make predictions
    prediction = model.predict(preprocessed_image)
    predicted_class = fish_classes[np.argmax(prediction)]  # Get the class with highest probability
    confidence = np.max(prediction)  # Confidence score

    # Display the prediction and confidence score in the second column
    with col2:
        st.markdown(f"<h3 style='color:gray;'>Predicted Fish Category:</h3> <h3 style='color:green;'>{predicted_class}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:gray;'>Confidence Score:</h3> <h3 style='color:green;'>{confidence:.2f}</h3>", unsafe_allow_html=True)
