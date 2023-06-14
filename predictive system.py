import numpy as np
import pickle

#loading the saved model
loaded_model = pickle.load(open('traineddp_model.sav', 'rb'))

input_data = (11,143,94,33,146,36.6,0.254,51)

#changing input data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
if prediction == 1:
  print('Patient is diabetic')
else:
  print("Patient is non-diabetic")