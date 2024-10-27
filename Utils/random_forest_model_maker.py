from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import precision_score

def predict(train, test, predictors, model):
    model.fit(train[predictors], train['target'])
    preds=model.predict_proba(test[predictors])[:,1]
    preds[preds>=0.6]=1
    preds[preds<0.6]=0
    preds=pd.Series(preds, index=test.index, name="Predictions")
    combined=pd.concat([test['target'], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions=[]
    
    for i in range(start, data.shape[0], step):
        train=data.iloc[0:i].copy()
        test=data.iloc[i:(i+step)].copy()
        predictions=predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

def random_forest_model_maker(dataset_rf):
    trainset_rf=dataset_rf.iloc[:-100]
    testset_rf=dataset_rf.iloc[-100:]
    predictors=['close', 'open', 'low', 'high', 'volume']
    dataset_rf['tomorrow']=dataset_rf['close'].shift(-1)
    dataset_rf['target']=(dataset_rf['tomorrow'] > dataset_rf['close']).astype(int)
    randomForest_model=RandomForestClassifier(n_estimators=300, min_samples_split=100, random_state=1)
    trainset_rf=dataset_rf.iloc[:-100]
    testset_rf=dataset_rf.iloc[-100:]
    predictors=['close', 'open', 'low', 'high', 'volume']
    randomForest_model.fit(trainset_rf[predictors], trainset_rf['target'])

    horizons=[12, 60, 120, 216, 288]
    new_predictors=[]

    for horizon in horizons:
        rolling_averages=dataset_rf.rolling(horizon).mean()
        ratio_column=f"Close_Ratio_{horizon}"
        dataset_rf[ratio_column]=dataset_rf['close']/rolling_averages['close']
    
        trend_column=f"Trend_{horizon}"
        dataset_rf[trend_column]=dataset_rf.shift(1).rolling(horizon).sum()['target']
    
        new_predictors+=[ratio_column, trend_column]
        dataset_rf=dataset_rf.dropna()
    
    predictions=backtest(dataset_rf.iloc[dataset_rf.shape[0]-105116:, :], randomForest_model, new_predictors)

    return precision_score(predictions['target'], predictions['Predictions'])

    
    