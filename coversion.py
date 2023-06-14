import joblib
import skl2onnx
import onnx
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load('traineddp_model.sav')
initial_type = [('float_input',FloatTensorType([None,8]))]

onnx_model = skl2onnx.convert.convert_sklearn(model, initial_types=initial_type)
onnx.save_model(onnx_model,'traineddp_model.onnx')