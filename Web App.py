import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open('traineddp_model.sav', 'rb'))

# Creating a function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'Patient is non-diabetic'
    else:
        return 'Patient is diabetic'

def main():
    st.title('Diabetes Prediction Web App')

    # Getting the input data from the user
    pregnancies = st.text_input('Number of Pregnancies:')
    glucose = st.text_input('Glucose Level:')
    blood_pressure = st.text_input('Blood Pressure value:')
    skin_thickness = st.text_input('Skin Thickness value:')
    insulin = st.text_input('Insulin level:')
    bmi = st.text_input('BMI value:')
    diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function value:')
    age = st.text_input('Age of the woman:')
    submit_button = st.button('Diabetes Test Result')

    # Validating input data
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
