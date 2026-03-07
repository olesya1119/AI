from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import uvicorn

app = FastAPI()
model = None


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


def main():
    global model

    # загрузка данных
    X, y = load_iris(return_X_y=True)

    # обучение модели
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # запуск сервера
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()