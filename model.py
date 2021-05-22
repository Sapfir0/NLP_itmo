import numpy as np
import pandas as pd
import pymorphy2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Model:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъыьэюя .'

    def normalize_text_with_morph(self, x):
        x = x.lower().replace("ё", "е")
        words = ''.join([[" ", i][i in self.alphabet] for i in x]).lower().split()
        return ' '.join([self.morph.parse(w)[0].normal_form for w in words])

    def prepare_test_train(self, train, test):
        train["message_a"] = train["message_a"].apply(self.normalize_text_with_morph)
        train["message_b"] = train["message_b"].apply(self.normalize_text_with_morph)
        test["message_a"] = test["message_a"].apply(self.normalize_text_with_morph)
        test["message_b"] = test["message_b"].apply(self.normalize_text_with_morph)

        return train[['message_a', 'message_b']], test[['message_a', 'message_b']]

    def _fit_predict(self, train, test):
        X, test_fixed = self.prepare_test_train(train, test)
        y = train['target']
        params = {
            'dictionaries': [
                {
                    'dictionary_id': 'Word',
                    'max_dictionary_size': '50000',
                    "occurrence_lower_bound" : "2",
                }
            ],
            'feature_calcers': [
                'BoW:top_tokens_count=10000'
            ]
        }
        model = CatBoostClassifier(
            verbose=100, 
            loss_function='CrossEntropy', 
            eval_metric='Accuracy',
            task_type='GPU',
            **params
        )
        learn_pool = Pool(
            X, y,
            text_features=['message_a', 'message_b'],
            feature_names=['message_a', 'message_b'],
        )
        test_pool = Pool(
            test_fixed,
            text_features=['message_a', 'message_b'],
            feature_names=['message_a', 'message_b'],
        )

        model.fit(learn_pool)
        predict = model.predict(test_pool)
        return pd.DataFrame(np.round(np.clip(predict, 0, 1)), columns=["target"])

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
