import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def importing_data(path):
    df=pd.read_csv(path)
    return df

def cleaning_data(df):
    df['tomorrow']=df['close'].shift(-1)
    df['target']=(df['tomorrow'] > df['close']).astype(int)
    df=df.drop(df.index[-1])
    del df['tomorrow']
    return df

def data_preprocessing(path):
    df=importing_data(path)
    df=cleaning_data(df)

    horizons=[12, 60, 120, 216, 288]
    new_predictors=[]

    for horizon in horizons:
        rolling_averages=df.rolling(horizon).mean()
        ratio_column=f"Close_Ratio_{horizon}"
        df[ratio_column]=df['close']/rolling_averages['close']
    
        trend_column=f"Trend_{horizon}"
        df[trend_column]=df.shift(1).rolling(horizon).sum()['target']
    
        new_predictors+=[ratio_column, trend_column]

    df=df.dropna()
    dataset=df.iloc[:, 0:df.shape[1]].values

    scaler=MinMaxScaler(feature_range=(0, 1))
    dataset_scaled=scaler.fit_transform(dataset)
    training_set, test_and_validation_set=train_test_split(dataset_scaled, test_size=0.01, shuffle=False)
    validation_set, test_set=train_test_split(test_and_validation_set, test_size=0.5, shuffle=False)
    X_train=[]
    y_train=[]
    for i in range(12, len(training_set)):
        X_train.append(training_set[i-12:i, :])
        y_train.append(training_set[i, 5])
    X_train, y_train=np.array(X_train), np.array(y_train)
    X_train=np.reshape(X_train, newshape=(X_train.shape[0], X_train.shape[1], df.shape[1]))

    X_validation=[]
    y_validation=[]
    for i in range(12, len(validation_set)):
        X_validation.append(validation_set[i-12:i, :])
        y_validation.append(validation_set[i, 5])
    X_validation, y_validation=np.array(X_validation), np.array(y_validation)
    X_validation=np.reshape(X_validation, newshape=(X_validation.shape[0], X_validation.shape[1], df.shape[1]))

    test_inputs=test_and_validation_set[len(test_and_validation_set)-len(test_set):]
    X_test=[]
    y_test=[]
    for i in range(12, len(test_inputs)):
        X_test.append(test_inputs[i-12:i, :])
        y_test.append(test_inputs[i, 5])
    X_test, y_test=np.array(X_test), np.array(y_test)
    X_test=np.reshape(X_test, newshape=(X_test.shape[0], X_test.shape[1], df.shape[1]))

    return X_train, y_train, X_validation, y_validation, X_test, y_test

