# Shipping Optimization Challenge my solution

## how to run

### RUN

```shell script
python src/prophet_v1.py
```

### My environment

#### Python
- Python 3.7.9
- conda
- library(requirements.txt)

#### Computer

- Ubuntu 20.04
- CPU: i9 9900K
- Memory: 16 * 2 (GiB)

## solution

### Features

Since there is no data for days when there was no trade, I have added such data that such days would also be zero. (Adding it increased the score compared to not adding it.)

I didn't do features engineering just to prepare the data.

### Modeling
I did the modeling based on prophet.

#### parmaeters

I used the data about holidays in the UK to train.

```python
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
```

#### What didn't work
- LGBM modeling(ex. https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50)
- complex parameters
- features(on Prophet)
