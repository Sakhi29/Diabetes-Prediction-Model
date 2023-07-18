import numpy as np
import joblib
import streamlit as st
import onnx
import onnxruntime as ort
from skl2onnx.common.data_types import FloatTensorType
import skl2onnx

def load_model():
    return joblib.load('traineddp_model.sav')

def convert_to_onnx(sklearn_model):
    initial_type = [('float_input', FloatTensorType([None, 8]))]
    onnx_model = skl2onnx.convert.convert_sklearn(sklearn_model, initial_types=initial_type)
    return onnx_model

def save_onnx_model(onnx_model):
    onnx.save_model(onnx_model, 'traineddp_model.onnx')

def diabetes_prediction(input_data):
    session = ort.InferenceSession('traineddp_model.onnx')
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1).astype(np.float32)

    prediction = session.run([output_name], {input_name: input_data_reshaped})
    if prediction[0][0] == 0:
        return 'Patient is non-diabetic'
    else:
        return 'Patient is diabetic'

def main():
    model = load_model()
    onnx_model = convert_to_onnx(model)
    save_onnx_model(onnx_model)

    st.title('Diabetes Prediction Web App')

    pregnancies = st.text_input('Number of Pregnancies:')
    glucose = st.text_input('Glucose Level:')
    blood_pressure = st.text_input('Blood Pressure value:')
    skin_thickness = st.text_input('Skin Thickness value:')
    insulin = st.text_input('Insulin level:')
    bmi = st.text_input('BMI value:')
    diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function value:')
    age = st.text_input('Age of the woman:')
    submit_button = st.button('Diabetes Test Result')

    error_message = ''
    if submit_button:
        if not all([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]):
            error_message = 'Please provide values for all input fields.'

    if error_message:
        st.error(error_message)
    elif submit_button:
        diagnosis = diabetes_prediction([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
        st.success(diagnosis)

if __name__ == '__main__':
    main()
