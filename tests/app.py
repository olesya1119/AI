from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# обучаем модель
X, y = load_iris(return_X_y=True)
model = LogisticRegression()
model.fit(X, y)

app = FastAPI()


# схема входных данных (валидация)
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    return {"prediction": int(prediction)}