import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def predict(testing_image):
    model = load_model('best_model_ML.h5')

    image = Image.open(testing_image).convert('RGB')
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image.reshape(1, 224, 224, 3)

    result = model.predict(image)
    result = np.argmax(result, axis=-1)

    if result == 0:
        return "The bee presents a varroa."
    elif result == 1:
        return "The bee is healthy."



def main():
    st.markdown(
        """<div style="text-align: center; font-weight: bold; font-size: 40px;">Varroa Detection</div>""",
        unsafe_allow_html=True
    )

    image_healthy = "pictures/healthy.jpeg"
    image_varroa = "pictures/varroa.jpeg"

    col1, col2 = st.columns(2)

    with col1:
        st.image(image_healthy, caption="Healthy bee", use_column_width=True)

    with col2:
        st.image(image_varroa, caption="Bee with varroa", use_column_width=True)

    st.markdown(
        """<div style="text-align: center; font-weight: bold; font-size: 24px;">This project will predict whether a bee has varroa mites using bee images. You can upload your own picture directly on the app.</div>""",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    image = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])


    if image is not None:
        # to view uploaded image
        st.image(Image.open(image))
        # Prediction
        if st.button('Result', help='Prediction'):
            prediction_result = predict(image)

            # Change background color to red if there is a varroa on the bee
            if "The bee presents a varroa." in prediction_result:
                st.error(prediction_result)
            else:
                st.success(prediction_result)


if __name__ == '__main__':
    main()