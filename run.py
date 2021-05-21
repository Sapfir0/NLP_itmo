import pandas as pd
from sklearn.metrics import accuracy_score

dataset = pd.DataFrame([["Первый текст для примера", "Второй текст для примера", 1],
                        ["Третий текст для примера", "Четвертый текст для примера", 0]], columns=["message_a", "message_b", "target"])

train = dataset
test = dataset[["message_a", "message_b"]]

true = [dataset[["target"]]] * 5

from model import Model

model = Model()
pred = model.fit_predict(train, test, train, test, train, test, train, test, train, test)

for i in range(5):
    print(f"Task_{i}:", accuracy_score(true[i], pred[i]))