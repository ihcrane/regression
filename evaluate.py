import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.linear_model import LinearRegression

def regression_errors(df, target, feature, pred_table=False):
    
    '''
    This function takes a dataframe, the target variable and the feature(s). It determines the baseline
    for the data. It then fits a LinearRegression model to the data and determines the SSE, MSE, RMSE,
    ESS, TSS and R^2. It returns all of the metrics in a data frame.
    '''
    
    # creating a preds data frame and determining the baseline predictions
    preds = df[feature]
    preds[target] = df[[target]]
    preds['baseline_preds'] = round(train[target].mean(), 2)
    
    # creating and fitting to a LinearRegression model then making predictions from the model
    model = LinearRegression().fit(train[feature], train[[target]])
    preds['yhat']= model.predict(train[feature])
    
    # determining the residuals of the baseline and predictions
    preds['baseline_res'] = preds['baseline_preds'] - preds[target]
    preds['yhat_res'] = preds['yhat'] - preds[target]
    
    # squaring the residuals
    preds['baseline_res_squared'] = preds['baseline_res'] ** 2
    preds['yhat_res_squared'] = preds['yhat_res'] ** 2
    
    # summing the squared residuals to get the SSE
    sse_baseline = preds['baseline_res_squared'].sum()
    sse_yhat = preds['yhat_res_squared'].sum()
    
    # dividing summed residuals by the number of predicitons to get the MSE
    mse_baseline = sse_baseline/len(preds)
    mse_yhat = sse_yhat/len(preds)
    
    # square rooting the MSE to get the RMSE
    rmse_baseline = sqrt(mse_baseline)
    rmse_yhat = sqrt(mse_yhat)
    
    # subtracting the baseline predictions from the predictions made by the model then squaring them
    preds['yhat_mean_res'] = preds['yhat'] - preds['baseline_preds']
    preds['yhat_mean_res_squared'] = preds['yhat_mean_res'] ** 2
    
    # summing the squared predictions to get the ESS
    ess_baseline = 0
    ess_yhat = preds['yhat_mean_res_squared'].sum()
    
    # adding the SSE and ESS to get the TSS
    tss_baseline = sse_baseline + ess_baseline
    tss_yhat = sse_yhat + ess_yhat
    
    # creating a DataFrame of the types of metrics
    metrics = pd.DataFrame(np.array(['SSE','MSE','RMSE','ESS','TSS', 'R^2']), columns=['metric'])
    
    # add the metrics determined to the DataFrame
    metrics['baseline'] = [sse_baseline, mse_baseline, rmse_baseline, 
                      ess_baseline, tss_baseline, (ess_baseline/tss_baseline)]
    metrics['yhat'] = [sse_yhat, mse_yhat, rmse_yhat, ess_yhat, 
                  tss_yhat, (ess_yhat/tss_yhat)]
    
    # returning metrics and predicition table
    if pred_table == True:
        return metrics, preds
    else:
        return metrics


def better_than_baseline(preds, target):

    '''
    This fucntion is to determine if the predictions are better than the baseline or not.
    '''

    baseline = mean_squared_error(preds[target], preds['baseline_preds'], squared=False)
    model = mean_squared_error(preds[target], preds['yhat'], squared=False)
    diff = baseline - model
    
    if diff > 0:
        return True
    else: 
        return False

def plot_residuals(df, y, yhat):

    '''
    This function plots the target value and the the predicted value in a scatter plot.
    '''
    
    plt.scatter(x=y, y=yhat, data=df.sample(2000))
    plt.xlabel(y)
    plt.ylabel(yhat)
    plt.show()
    