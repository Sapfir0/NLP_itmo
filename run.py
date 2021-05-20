from model import Model
import pandas as pd

m = Model()

trainDict = {'message_a': ['sfasfa'], 'message_b': ['fafqwr11r'], 'target': [1]}
train = pd.DataFrame(data=trainDict)

testDict = {'message_a': ['sfafafagasg'], 'message_b': ['fafqwr11r'], 'target': [0]}
test = pd.DataFrame(data=testDict)


m._fit_predict(train, test)