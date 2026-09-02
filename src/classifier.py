import joblib


class Classifier:
    def __init__(self):
        self.__path = "toxic_linear_svc.joblib"
        self.__package = joblib.load(self.__path)
        self.__pipeline = self.__package["pipeline"]

    def __predict__(self, x) -> dict:
        return {
            "is_toxic": bool(self.__pipeline.predict([x])[0])
        }
    def predict(self, x) -> dict:
        return self.__predict__(x)
