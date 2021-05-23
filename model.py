import numpy as np
import pandas as pd
import pymorphy2
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesRegressor, VotingRegressor, RandomForestRegressor
from tensorflow import keras

class Model:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.alphabet = 'abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъыьэюя .'

    def to_binary(self, predicted):
        return np.round(np.clip(predicted, 0, 1))

    def normalize_text_with_morph(self, x):
        x = x.lower().replace("ё", "е")
        words = ''.join([[" ", i][i in self.alphabet] for i in x]).lower().split()
        return ' '.join([self.morph.parse(w)[0].normal_form for w in words])

    def normalize_data(self, train, test):
        train["message_a"] = train["message_a"].apply(self.normalize_text_with_morph)
        train["message_b"] = train["message_b"].apply(self.normalize_text_with_morph)
        test["message_a"] = test["message_a"].apply(self.normalize_text_with_morph)
        test["message_b"] = test["message_b"].apply(self.normalize_text_with_morph)

        return train[['message_a', 'message_b']], test[['message_a', 'message_b']]


    def vectorize_data(self, train, test):
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
        return  _train, _test


    def _get_catboost_model(self):
        params = {
            'tokenizers': [
                {
                    'tokenizer_id': 'Sense',
                    'separator_type': 'BySense',
                    'lowercasing': 'True',
                    'token_types':['Word', 'Number', 'SentenceBreak'],
                    'sub_tokens_policy':'SeveralTokens'
                }      
            ],
            'dictionaries': [
                {
                    'dictionary_id': 'Word',
                    'max_dictionary_size': '50000',
                    "occurrence_lower_bound" : "1",
                }
            ],
            'feature_calcers': [
                'BoW:top_tokens_count=10000',
            ]
        }
        model = CatBoostClassifier(
            verbose=False, 
            eval_metric='Accuracy',
            task_type='GPU',
            **params
        )
        return model

    def get_test_pool(self, test):
        return Pool(
            test,
            text_features=['message_a', 'message_b'],
            feature_names=['message_a', 'message_b'],
        )

    def predict_catboost(self, model, X_train, X_test, y_train):
        learn_pool = Pool(
            X_train, y_train,
            text_features=['message_a', 'message_b'],
            feature_names=['message_a', 'message_b'],
        )

        test_pool = self.get_test_pool(X_test)

        model.fit(learn_pool)
        predict = model.predict(test_pool)
        return predict

    def get_LTSM_model(self, input_shape):
        model = keras.Sequential([
            keras.layers.Input(shape=input_shape),
            keras.layers.LSTM(40, return_sequences=True),
            keras.layers.LSTM(40),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(1, activation='hard_sigmoid'),
        ])
        model.summary()
        model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
        return model

    def prepare_LTSM_data(self, train, test):
        X, vtest = self.vectorize_data(train, test)
        X = np.reshape(X, (X.shape[0], 2, X.shape[1]//2))
        vtest = np.reshape(vtest, (vtest.shape[0], 2, vtest.shape[1]//2))
        return X, vtest

    def get_best_model(self, train, test):
        scores = []
        X, vtest = self.prepare_LTSM_data(train, test)
        input_shape = (X.shape[-2], X.shape[-1])

        models = [ 
            {
                'model': self._get_catboost_model(), 
                'prepareDataCallback': self.normalize_data, 
                'customFeat': self.predict_catboost,
                'testDataGetter': self.get_test_pool
            },
            {
                'model': self.get_LTSM_model(input_shape), 
                'prepareDataCallback': self.prepare_LTSM_data,
                 
            }
        ]
        y = train['target']

        for modelInfo in models:
            vtrain, vtest = modelInfo['prepareDataCallback'](train, test)
            X_train, X_test, y_train, y_test = train_test_split(vtrain, y, test_size=0.2, random_state=42)

            model = modelInfo['model']
            if 'customFeat' in modelInfo:
                predicted = modelInfo['customFeat'](model, X_train, X_test, y_train) 
            else:
                model.fit(X_train, y_train)
                predicted = model.predict(X_test)
            
            score = accuracy_score(y_test, self.to_binary(predicted))
            scores.append(score)
        
        print(scores)
        bestIndex = scores.index(max(scores))
        # return catboost
        return models[bestIndex]

    def get_test_data_for_model(self, model_desc, train, test):
        _, _test = model_desc['prepareDataCallback'](train, test)

        if 'testDataGetter' in model_desc:
            return model_desc['testDataGetter'](_test)
        return _test

    def _fit_predict(self, train, test):
        model_desc = self.get_best_model(train, test)
        print(model_desc)

        _test = self.get_test_data_for_model(model_desc, train, test)

        predict = model_desc['model'].predict(_test)
        return pd.DataFrame(self.to_binary(predict), columns=["target"])

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
