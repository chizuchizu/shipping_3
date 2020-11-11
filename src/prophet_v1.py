from fbprophet import Prophet
from fbprophet.make_holidays import make_holidays_df
from sklearn.metrics import mean_squared_error as mse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import pickle

from src.load_base_data import preprocessed_data


def load_data():
    data = preprocessed_data()[["send_timestamp", "target", "train", "shipping_company"]]
    data = data[data["train"]].drop(columns="train")
    data = data.rename(columns={"send_timestamp": "ds", "target": "y"})
    return data


def plot_model(m, forecast, pars, mode):
    fig = m.plot(forecast)
    ax1 = fig.add_subplot(111)
    ax1.set_title(f"Forecast {mode}_{pars}", fontsize=16)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Number", fontsize=12)

    fig2 = m.plot_components(forecast)

    plt.show()


def metric(pred, label):
    max_observed = max(label)
    return np.exp(- (np.sqrt(mse(pred, label)) / max_observed))


train = load_data()
best_score = 0
use_all_data = True
debug = True
rand = np.random.randint(0, 1000000)
experiment_name = f"debug_{rand}" if debug else f"{rand}"
year_list = [2019, 2020]
holidays = make_holidays_df(year_list=year_list, country='UK')
params = {
    "growth": "logistic",
    # "changepoint_prior_scale": 0.05,
    # "seasonality_prior_scale": 10,
    # "mcmc_samples": 0,
    "seasonality_mode": "multiplicative",
    "daily_seasonality": True,
    "weekly_seasonality": True,
    # "changepoints": ["2020-03-23", "2020-05-11"],
    "holidays": holidays
    #  "prophet_pos": multiplicative
    # "likelihood": "NegBinomial"
}


def run_model():
    all_score = 0
    all_metric = 0
    for x in ["SC1", "SC2", "SC3"]:
        if not use_all_data:
            if x == "SC2":
                use_data = train.loc[train["shipping_company"] == x, ["ds", "y"]]
                periods = 61
            else:
                use_data = train.loc[train["shipping_company"] == x, ["ds", "y"]]
                use_data = use_data.loc[pd.to_datetime(use_data["ds"]) < pd.to_datetime("2019-12-01")]
                periods = 61 + 14 + 31 + 30 + 31 + 29 + 31 + 31
        else:
            use_data = train.loc[train["shipping_company"] == x, ["ds", "y"]]
            periods = 61

        # mlflow.set_experiment(experiment_name)
        mlflow.set_experiment(x)

        # with mlflow.start_run(run_name=f"{x}"):
        with mlflow.start_run(run_name=f"{rand}"):
            model = build_model()
            if params["growth"] == "logistic":
                use_data["cap"] = use_data["y"].max() * 1.2
            model.fit(use_data)
            future = model.make_future_dataframe(periods=periods)

            # future["lockdown"] = future["ds"].apply(is_lockdown)
            # future["normal"] = ~future["ds"].apply(is_lockdown)

            if params["growth"] == "logistic":
                future["cap"] = use_data["y"].max() * 1.2

            forecast = model.predict(future)

            oof = forecast["yhat"].iloc[:-periods]
            val_y = use_data["y"]

            # score = np.exp(-np.sqrt(mse(oof, val_y)))
            score = np.sqrt(mse(oof, val_y))
            metric_ = metric(oof, val_y)

            print("score: ", score, metric_)
            all_score += score
            all_metric += metric_

            pred = forecast[["ds", "yhat"]]
            # if x == "SC2":
            pred = pred.iloc[-61:, :]

            pred["company"] = x

            # https://github.com/facebook/prophet/issues/725
            pkl_path = f"../models/{rand}_{x}_prophet.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(model, f)

            # plot_model(model, forecast, pars, x)
            fig1 = model.plot(forecast)
            plt.savefig("fig1.png")
            plt.clf()
            fig2 = model.plot_components(forecast)
            plt.savefig("fig2.png")
            plt.clf()

            if x == "SC1":
                data = pred
            else:
                data = pd.concat([data, pred])

            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("rmse", score)
            mlflow.log_metric("metric", metric_)
            mlflow.log_artifact("fig1.png")
            mlflow.log_artifact("fig2.png")
            # mlflow.log_param("ProphetPos", prophet_pos)

    all_score /= 3
    all_metric /= 3
    data.loc[data["yhat"] < 0, "yhat"] = 0
    sorted_data = data.sort_values(["ds", "company"])
    return all_score, all_metric, sorted_data["yhat"], data


def build_model():
    m = Prophet(**params)

    m = m.add_seasonality(
        name='weekly',
        period=7,
        fourier_order=15)

    m = m.add_seasonality(
        name='monthly',
        period=30.5,
        fourier_order=25)

    # m = m.add_seasonality(
    #     name='yearly',
    #     period=365.25,
    #     fourier_order=yseas)

    return m


score, metric_, forecast, data = run_model()

print(score, metric_)
# forecast[forecast < 0] = 0
forecast.to_csv(f"../outputs/{round(best_score, 5)}_{rand}_prophet.csv", index=False, header=False)

mlflow.set_experiment("all")
with mlflow.start_run(run_name=f"{rand}"):
    for k, v in params.items():
        mlflow.log_param(k, v)
    mlflow.log_metric("rmse", score)
    mlflow.log_metric("metric", metric_)
    mlflow.log_artifact(f"../outputs/{round(best_score, 5)}_{rand}_prophet.csv")

    for x in ["SC1", "SC2", "SC3"]:
        memo = data.loc[data["company"] == x]
        for i, row in enumerate(memo.itertuples()):
            mlflow.log_metric(key=x, value=row.yhat, step=i)
