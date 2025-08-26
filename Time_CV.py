import statsmodels.api as sm
import numpy as np
from copy import copy
class temporal_cross_validation():
    def __init__(self, endog, exog, val_size, train_size, test_size):
        self.endog = endog
        self.exog = exog
        self.val_size   = val_size
        self.train_size = train_size
        self.test_size  = test_size
        self.len = len(endog)
        self.idx = 0
        self.mode = None
    def get_val(self):
        return self.endog[self.len - self.val_size: ], self.exog[self.len - self.val_size: ]
    
    def set_mode(self, mode):
        if mode == "SLIDING" or mode == "PERSISTENT":
            self.mode = copy(mode)
        else:
            raise ValueError("Unknown mode, available modes: {SLIDING, PERSISTENT}.")

    def reset(self):
        self.idx = 0

    def __len__(self):
        return self.len - self.train_size - self.val_size

    def get_data(self):
        return self.endog, self.exog    
    
    def get_train_test(self):
        start = self.idx
        if self.mode is None or self.mode == "SLIDING":
            start = self.idx
        elif self.mode == "PERSISTENT":
            start = 0
        if self.idx + self.train_size <= self.len - self.val_size:
            # Start, Train End, Test End
            return start, self.idx + self.train_size, self.idx + self.train_size + self.test_size
        else:
            # Validation must be separated from train
            return None, None, None
            

        
    def iterate(self, step= 1):
        self.idx += max(step, 1)


def apply_out_of_domain(fitted_model, forecast_steps, ENDOG, EXOG=None, return_train_preds=False):
    model = fitted_model.model.clone(
        endog= ENDOG,
        exog= EXOG,
    )
    param_dict = dict(zip(fitted_model.param_names, fitted_model.params))
    with model.fix_params(param_dict):
        refitted_model = model.fit()
    if forecast_steps == 0:
        return None, refitted_model.predict()
    if return_train_preds:
        return refitted_model.forecast(forecast_steps), refitted_model.predict()
    return refitted_model.forecast(forecast_steps), None

def mape(y_true, y_pred, zero_division=1e-7):
    return np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), zero_division))))

def generate_lags(
        data: np.ndarray, # shape (n_samples, n_features)
        max_lags: int,
        filler = np.nan
    ) -> np.ndarray: # shape (n_samples, n_features, max_lags + 1)
    data_copy = np.concatenate((filler * np.ones((max_lags, data.shape[1])), data), axis=0)
    return np.flip(np.lib.stride_tricks.sliding_window_view(data_copy, max_lags + 1, axis=0), axis= len(data_copy.shape) - 1)

if __name__ == "__main__":
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import warnings
    import tqdm.auto as tqdm
    warnings.simplefilter('ignore', UserWarning)

    all_q = pd.read_excel("data\Данные по лучшим моделям.xlsx")
    br_q = pd.read_excel("data\Данные по лучшим моделям.xlsx", 1)
    all_q.dropna(inplace=True)
    br_q.dropna(inplace=True)
    br_lags = generate_lags(br_q.to_numpy(), 12)[:, :, ::-1]
    br_exog_data = np.concatenate(
        (
            br_lags[:, 0, (2,3,8,9,10,11,12)],
            br_lags[:, 1, (1, 2, 4, 5)],
            br_lags[:, 3, (10, 11)],
            br_lags[:, 4, (11)].reshape(-1, 1),
            br_lags[:, 5, (4, 8, 9)],
        ),
        axis= 1
    )
    raw_data = pd.read_excel("data/Рождаемость_данные.xlsx")
    raw_data.set_index("month", inplace=True)
    

    ARIMA_params = (2, 1, 1)      # (p, d, q)
    SARIMA_params = (0, 1, 1, 12) # (P, D, Q, S)

    VAL_SIZE = 0
    TRAIN_SIZE = 60
    TEST_SIZE = 0

    MODE = "PERSISTENT"

    # SARIMAX treats exog variables same way as endog variables, so undifference is needed
    br_exog_data_unnormalized = np.zeros((br_exog_data.shape[0] + ARIMA_params[1], br_exog_data.shape[1]))
    br_exog_data_unnormalized[ARIMA_params[1]:]  += br_exog_data
    br_exog_data_unnormalized[:-ARIMA_params[1]] += br_exog_data

    max_lags = -1
    temporal_dataset = temporal_cross_validation(raw_data.to_numpy()[max_lags + 1:] / 100000, br_exog_data_unnormalized[max_lags + 1:], VAL_SIZE, TRAIN_SIZE, TEST_SIZE)
    new_index = raw_data.index[max_lags + 1:]

    temporal_dataset.set_mode(MODE)

    endog, exog = temporal_dataset.get_data()
    exog = None
    val, val_endog = temporal_dataset.get_val()

    start, train_end, test_end = temporal_dataset.get_train_test()

    test_values = []
    val_values = []

    reports = []
    models_params = []

    counter = 0
    pbar = tqdm.tqdm(total= len(temporal_dataset))

    while train_end is not None:
        model = sm.tsa.statespace.SARIMAX(
            endog= endog[start:train_end],
            exog= None if exog is None else exog[start:train_end], 
            order= ARIMA_params,
            seasonal_order= SARIMA_params,
        )
        fitted = model.fit(disp= False)

        # reports.append((counter, fitted.summary()))
        models_params.append({"params": dict(zip(fitted.param_names, fitted.params)), "start": start, "train_end": train_end})

        temporal_dataset.iterate()
        start, train_end, test_end = temporal_dataset.get_train_test()
        
        counter += 1
        pbar.update(1)
    del pbar
    import json

    with open(f"simple_{MODE.lower()}_models.json", "w") as f:
        result = {
            "ARIMA_params": ARIMA_params,
            "SARIMA_params": SARIMA_params,
            "VAL_SIZE": VAL_SIZE,
            "TRAIN_SIZE": TRAIN_SIZE,
            "TEST_SIZE": TEST_SIZE,
            # "reports": list(map(lambda x: str(x), reports)),
            "models": models_params,
            "index": new_index.to_list()
        }
        json.dump(result, f, indent=4)
    with open(f"simple_{MODE.lower()}_models.json", 'r') as f:
        reprod_sliding_models = json.load(f)
        models_params = reprod_sliding_models['models']
        new_index = reprod_sliding_models['index']
    desired_val_sizes = [12, 24, 36]

    val_values = []
    for _ in desired_val_sizes:
        val_values.append([])


    for elem in tqdm.tqdm(models_params):
        refit_model = fitted.model.clone(
            endog,
            exog,
        )
        with refit_model.fix_params(elem["params"]):
            refitted_model = refit_model.fit()
        for i in range(len(desired_val_sizes)):
            val_values[i].append((elem["train_end"], refitted_model.predict()[-desired_val_sizes[i]:]))
            
    for i, VAL_SIZE in enumerate(desired_val_sizes):
        idxes = []
        mapes = []
        for elem in val_values[i][:-VAL_SIZE]:
            mapes.append(mape(endog[-VAL_SIZE:].flatten(), elem[1]))
            idxes.append(new_index[elem[0] - 9])

        plt.figure()
        a = pd.DataFrame(mapes, index=idxes, columns=["MAPE"])
        a.plot(xlabel='Отсечка данных с лагом в 9 месяцев вперёд', ylabel='MAPE')
        a.to_excel(f"simple_{MODE.lower()}_mapes_{VAL_SIZE}.xlsx")
        plt.show()