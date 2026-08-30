import joblib


class Classifier:
    def __init__(self):
        self.__path = "toxic_linear_svc.joblib"
        self.__package = joblib.load(self.__path)
        self.__pipeline = self.__package["pipeline"]
        self.__f1_macro = self.__package["f1_macro"]
    def __predict__(self, x) -> dict:
        return {
            "is_toxic": bool(self.__pipeline.predict([x])[0]),
            "confidence": self.__f1_macro
        }
    def predict(self, x) -> dict:
        return self.__predict__(x)