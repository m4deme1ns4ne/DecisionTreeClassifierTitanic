from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from loguru import logger


class PredictionsInput(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_Q: int
    Embarked_S: int

app = FastAPI(
    title="API предсказания выживания на Титанике",
    description="API для предсказания выживания пассажиров на Титанике на основе их данных.",
    version="1.0.0"
)

try:
    with open('final_model.joblib', 'rb') as file:
        model = joblib.load(file)
    logger.info("Модель успешно загружена.")
except Exception as err:
    logger.error(f"Не удалось загрузить модель: {err}")
    raise HTTPException(status_code=500, detail="Ошибка загрузки модели.")

requests_count = 0


@app.get("/stats", summary="Получить статистику запросов")
async def stats():
    """
    Возвращает количество запросов на предсказание, сделанных к API.
    """
    return {"requests_count": requests_count}


@app.get("/health", summary="Проверка состояния")
async def health_check():
    """
    Проверяет, работает ли API.
    """
    return {"status": "ok"}


@app.post("/predict_model", summary="Сделать предсказание выживания")
def predict_model(input_data: PredictionsInput):
    """
    Предсказывает, выжил бы пассажир на Титанике, на основе входных данных.

    - **Pclass**: Класс пассажира (1, 2, 3)
    - **Sex**: 0 для мужчин, 1 для женщин
    - **Age**: Возраст пассажира
    - **SibSp**: Количество братьев, сестер или супругов на борту
    - **Parch**: Количество родителей или детей на борту
    - **Fare**: Стоимость билета
    - **Embarked_Q**: 1, если посадка в Квинстауне, иначе 0
    - **Embarked_S**: 1, если посадка в Саутгемптоне, иначе 0
    """
    global requests_count
    requests_count += 1

    try:
        new_data = pd.DataFrame([[
            input_data.Pclass,
            input_data.Sex,
            input_data.Age,
            input_data.SibSp,
            input_data.Parch,
            input_data.Fare,
            input_data.Embarked_Q,
            input_data.Embarked_S
        ]], columns=model.feature_names_in_)
        
        predictions = model.predict(new_data)

        result = "Survived" if predictions[0] == 1 else "Not survived"

        return {'predictions': result}
    except ValueError as err:
        logger.error(f"Ошибка предсказания: {err}")
        raise HTTPException(status_code=400, detail="Неверные входные данные.")
    except Exception as err:
        logger.error(f"Неизвестная ошибка: {err}")
        raise HTTPException(status_code=500, detail="Ошибка предсказания.")

def main():
    logger.info("API запущен.")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
