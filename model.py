import numpy as np
import pandas as pd
import pymorphy2
from catboost import CatBoostRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъыьэюя '

    def normalize_text_with_morph(self, x):
        x = x.lower().replace("ё", "е")
        words = ''.join([[" ", i][i in self.alphabet] for i in x]).lower().split()
        return ' '.join([self.morph.parse(w)[0].normal_form for w in words])


    def getModel(self, _train, trueData):
        X_train, X_test, y_train, y_test = train_test_split(_train, trueData, test_size=0.2, random_state=42)
        print(y_train)
        models = [ 
            {
                'model': CatBoostRegressor(
                    task_type='GPU',
                    random_state=42, 
                    allow_const_label=True
                ),
                'fitParams': {'eval_set': (X_test, y_test), 'use_best_model': True}
            },
            {
                'model': ExtraTreesRegressor(n_jobs=-1),
                'fitParams': {}
            }
        ]

        scores = []

        for modelInfo in models:
            model = modelInfo['model']
            
            # model.fit(_train, trueData, **modelInfo['fitParams'])
            model.fit(X_train, y_train, **modelInfo['fitParams'])

            predictedData = model.predict(X_test)
            score = accuracy_score(y_test, np.round(np.clip(predictedData, 0, 1)))
            scores.append(score)

        bestIndex = scores.index(max(scores))
        print(scores)

        return models[bestIndex]['model']



    def _fit_predict(self, train, test):
        vectorizer_a = TfidfVectorizer(max_features=1000)
        vectorizer_b = TfidfVectorizer(max_features=1000)

        train["message_a"] = train["message_a"].apply(self.normalize_text_with_morph)
        train["message_b"] = train["message_b"].apply(self.normalize_text_with_morph)
        train_a = vectorizer_a.fit_transform(train["message_a"]).toarray()
        train_b = vectorizer_b.fit_transform(train["message_b"]).toarray()
        _train = np.hstack([train_a, train_b])

        test["message_a"] = test["message_a"].apply(self.normalize_text_with_morph)
        test["message_b"] = test["message_b"].apply(self.normalize_text_with_morph)
        test_a = vectorizer_a.transform(test["message_a"]).toarray()
        test_b = vectorizer_b.transform(test["message_b"]).toarray()
        _test = np.hstack([test_a, test_b])


        model = self.getModel(_train, train["target"])
        print(model)
        model.fit(_train, train["target"])
        predicted = model.predict(_test)
        

        return pd.DataFrame(np.round(np.clip(predicted, 0, 1)), columns=["target"])

    def fit_predict(self,
                    train_1, test_1,
                    train_2, test_2,
                    train_3, test_3,
                    train_4, test_4,
                    train_5, test_5):
        predicted_1 = self._fit_predict(train_1, test_1)
        predicted_2 = self._fit_predict(train_2, test_2)
        predicted_3 = self._fit_predict(train_3, test_3)
        predicted_4 = self._fit_predict(train_4, test_4)
        predicted_5 = self._fit_predict(train_5, test_5)
        return [predicted_1, predicted_2, predicted_3, predicted_4, predicted_5]

