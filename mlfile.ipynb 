import pandas as pd
import catboost

import numpy as np
from sklearn.model_selection import KFold
from copy import deepcopy
from tqdm.notebook import tqdm
import os
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

train = pd.read_csv("/Users/mikhailtaipov/Downloads/ef-msu-master-comp3/train.csv")
test = pd.read_csv("/Users/mikhailtaipov/Downloads/ef-msu-master-comp3/test.csv")
sample=pd.read_csv("/Users/mikhailtaipov/Downloads/ef-msu-master-comp3/sample_submission.csv")
test_ids = test["id"]
def mape(y,p):
    return np.nanmean(np.divide(np.abs((y - p)), np.abs(y)))
train["checkout_price"]=train["checkout_price"]-train["checkout_price"].mean()
train["checkout_price"]=train["checkout_price"]/(train["checkout_price"].std())
train["base_price"]=train["base_price"]-train["base_price"].mean()
train["base_price"]=train["base_price"]/(train["base_price"].std())
 
train["rating"]=train["rating"]-train["rating"].mean()
train["rating"]=train["rating"]/(train["rating"].std())
train["op_area"]= (np.round(train["op_area"])).astype(int)
train["relprice"]=train["checkout_price"]/train["base_price"]
 
train.head()
test["checkout_price"]=test["checkout_price"]-test["checkout_price"].mean()
test["checkout_price"]=test["checkout_price"]/(test["checkout_price"].std())
test["base_price"]=test["base_price"]-test["base_price"].mean()
test["base_price"]=test["base_price"]/(test["base_price"].std())
 

test["rating"]=test["rating"]-test["rating"].mean()
test["rating"]=test["rating"]/(test["rating"].std())
test["relprice"]=test["checkout_price"]/test["base_price"]
 
test["op_area"]= (np.round(test["op_area"])).astype(int)
test.head()
def cv_and_predict(
    df_train,
    df_test,
    train_y,
    model,
    log_transform_target=True,
    do_scaling=False,
    n_splits=5,
    random_state=42,
    verbose=True,
):
    """
    Функция для кросс-валидации и предикта на тест

    :param df_train: Трейн-датафрейм
    :param df_test: Тест-датафрейм
    :param train_y: Ответы на трейн
    :param model: Модель, которую мы хотим учить
    :param log_transform_target: Делаем ли лог-трансформацию таргета при обучении
    :param do_scaling: Делаем ли скейлинг признаков
    :param n_splits: Количество сплитов для KFold
    :param random_state: random_state для KFold
    :param verbose: Делаем ли print'ы

    :return: pred_test: Предсказания на тест; oof_df: OOF предсказания
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # В датафрейме oof_df будут храниться настоящий таргет трейна и OOF предсказания на трейн.
    # Инициализируем prediction_oof нулями и будем заполнять предсказаниями в процессе валидации
    oof_df = pd.DataFrame()
    oof_df["target"] = train_y
    oof_df["prediction_oof"] = np.zeros(oof_df.shape[0])

    # Список с метриками по фолдам
    metrics = []

    # Предсказания на тест. Инициализируем нулями и будем заполнять предсказаниями в процессе валидации.
    # Наши предсказания будут усреднением n_splits моделей
    pred_test = np.zeros(df_test.shape[0])

    # Кросс-валидация
    for i, (train_index, valid_index) in enumerate(kf.split(df_train, train_y)):
        if verbose:
            print(f"fold_{i} started")

        X_train = df_train.loc[train_index]
        y_train = train_y.loc[train_index].values

        if do_scaling:
            scaler = StandardScaler()
            columns = X_train.columns
            X_train = scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train)
            X_train.columns = columns

        X_valid = df_train.loc[valid_index]
        y_valid = train_y.loc[valid_index].values

        if log_transform_target:
            y_train = np.log1p(y_train)
            y_valid = np.log1p(y_valid)

        if do_scaling:
            X_valid = scaler.transform(X_valid)
            X_valid = pd.DataFrame(X_valid)
            X_valid.columns = columns

        model_kf = deepcopy(model)

        model_kf.fit(
            X_train,
            y_train,
            plot=True,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=50,
        )

        if do_scaling:
            df_test_scaled = scaler.transform(df_test)
            df_test_scaled = pd.DataFrame(df_test_scaled)
            df_test_scaled.columns = columns
            prediction_kf = model_kf.predict(df_test_scaled)
        else:
            prediction_kf = model_kf.predict(df_test)

        if log_transform_target:
            prediction_kf = np.expm1(prediction_kf)

        pred_test += prediction_kf / n_splits

        prediction = model_kf.predict(X_valid)

        if log_transform_target:
            prediction = np.expm1(prediction)
            y_valid = np.expm1(y_valid)
        oof_df.loc[valid_index, "prediction_oof"] = prediction

        cur_metric = mape(y_valid, prediction)
        metrics.append(cur_metric)
        if verbose:
            print(f"metric_{i}: {cur_metric}")
            print()
            print("_" * 100)
            print()

    metric_OOF = mape(train_y, oof_df["prediction_oof"])

    if verbose:
        print(f"metric_OOF: {metric_OOF}")
        print(f"metric_AVG: {np.mean(metrics)}")
        print(f"metric_std: {np.std(metrics)}")
        print()
        print("*" * 100)
        print()

    return pred_test, oof_df, metric_OOF
cbs_reg = CatBoostRegressor(
    loss_function="LogLinQuantile",
    learning_rate=0.9,
    
    random_state=1337,
    thread_count=-1,
    num_trees=300,
    cat_features=["city_code", "region_code", "center_type", "category", "cuisine","homepage_featured","emailer_for_promotion"],
    verbose=200,depth=12,langevin=True,      
)
pred_test, oof_df, metric_OOF = cv_and_predict(train.drop(['num_orders', 'id','week'], axis=1),
                                               test.drop(['id','week'], axis=1), 
                                               train_y=train['num_orders'],  
    n_splits=5, model=cbs_reg)
submission = pd.DataFrame()
submission["id"] = test_ids
submission["num_orders"] = pred_test
submission
test.head()
submission.to_csv("/Users/mikhailtaipov/Downloads/submission_basefilThe45530027000038.csv", index=False)
