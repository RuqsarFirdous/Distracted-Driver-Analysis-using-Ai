import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage

#Load your trained model with caching
@st.cache_resource
def load_my_model():
    return load_model('best_model.keras')

model = load_my_model()

# Custom CSS for background only — no forced upload styling
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                    url('https://pub.mdpi-res.com/sustainability/sustainability-16-07642/article_deploy/html/images/sustainability-16-07642-g002.png?1725356804');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    [data-testid="stAppViewContainer"] * {
        color: #ffffff;
    }

    /* Optional: Outer uploader box only (neutral) */
    section[data-testid="stFileUploader"] {
        background: rgba(0, 0, 0, 0.4);
        border-radius: 12px;
        padding: 20px;
        border: 2px solid #ffffff;
    }

    /* Label text */
    section[data-testid="stFileUploader"] label {
        color: #ffffff;
        font-weight: bold;
    }

    /* Alerts */
    .stAlert {
        background: rgba(0,0,0,0.6) !important;
        border: 1px solid #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Title & description
st.title("🚗 Driver Distraction Detection")
st.write("Upload a driver image to check if they are **focused** or **distracted**.")

# Image uploader (default theme styling)
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Preprocessing
def load_and_preprocess(img, target_size=(200, 200)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Prediction
if uploaded_file is not None:
    img = PILImage.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img_array = load_and_preprocess(img)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    if predicted_class == 0:
        st.success("The driver is **FOCUSED** (c0).")
    else:
        st.warning(f"⚠️ The driver is **DISTRACTED** — detected class: c{predicted_class}.")

