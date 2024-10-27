from Utils.Data_mining import data_mining
from Utils.data_preprocessing import data_preprocessing, importing_data
from Utils.LSTM_classification_model_maker import create_lstm_classification_model
from Utils.LSTM_reg_model_maker import lstm_reg_model_maker
from Utils.random_forest_model_maker import random_forest_model_maker

#df=data_mining("ETH-USDT", "5m", "2020/04/09", "2023/08/11")
#df.to_csv("ETH-USDT_5m.csv")

dataset=importing_data('./Dataset/ETS-USDT_5m.csv')
X_train, y_train, X_validation, y_validation, X_test, y_test=data_preprocessing('./Dataset/ETH-USDT_5m.csv')

lstm_classification_model=create_lstm_classification_model(50, 0.2, X_train, y_train)
lstm_reg_model=lstm_reg_model_maker(50, 0.2, X_train, y_train)

lstm_reg_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_validation, y_validation))
lstm_classification_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_validation, y_validation))

calssification_score=lstm_classification_model.evaluate()
regression_score=lstm_reg_model.evaluate()
random_forest_model_score=random_forest_model_maker(dataset)
