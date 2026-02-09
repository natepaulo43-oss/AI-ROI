from .schemas import PredictionInput, PredictionOutput

def make_prediction(model, input_data: PredictionInput) -> PredictionOutput:
    prediction = model.predict([input_data.features])
    return PredictionOutput(prediction=float(prediction[0]))
