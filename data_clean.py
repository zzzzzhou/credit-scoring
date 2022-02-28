import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def clean(path):
    rawdata = pd.read_csv(path)
    rawdata = rawdata.drop(['ID', 'year'], axis=1)
    imp_fill0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    term = rawdata.loc[:, 'term'].values.reshape(-1,1) #change series into a columns of data to meet imputer requirement
    property_value = rawdata.loc[:, 'property_value'].values.reshape(-1, 1)
    rawdata['term'] = imp_fill0.fit_transform(term)
    rawdata['property_value'] = imp_fill0.fit_transform(property_value)
    # fill null with 0

    imp_fillother = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='other')
    loan_limit = rawdata.loc[:, 'loan_limit'].values.reshape(-1, 1)
    approv_in_adv = rawdata.loc[:, 'approv_in_adv'].values.reshape(-1, 1)
    loan_purpose = rawdata.loc[:, 'loan_purpose'].values.reshape(-1, 1)
    Neg_ammortization = rawdata.loc[:, 'Neg_ammortization'].values.reshape(-1, 1)
    age = rawdata.loc[:, 'age'].values.reshape(-1, 1)
    rawdata['loan_limit'] = imp_fillother.fit_transform(loan_limit)
    rawdata['approv_in_adv'] = imp_fillother.fit_transform(approv_in_adv)
    rawdata['loan_purpose'] = imp_fillother.fit_transform(loan_purpose)
    rawdata['Neg_ammortization'] = imp_fillother.fit_transform(Neg_ammortization)
    rawdata['age'] = imp_fillother.fit_transform(age)
    #fill null with 'other'

    middata = rawdata[
        ['loan_type', 'Credit_Worthiness', 'open_credit', 'business_or_commercial', 'interest_only', 'lump_sum_payment',
         'Secured_by']]
    enc = OrdinalEncoder(dtype=int)
    rawdata[
        ['loan_type', 'Credit_Worthiness', 'open_credit', 'business_or_commercial', 'interest_only', 'lump_sum_payment',
         'Secured_by']] = enc.fit_transform(middata)
    # change text data into 1/0

    rawdata[['status_interestrate', 'status_upcharge', 'status_income', 'status_ltv']] = 0 #creat 4 new colomns and set to 0
    rawdata.loc[pd.isnull(rawdata['rate_of_interest']) == True, 'status_interestrate'] = 1
    rawdata.loc[pd.isnull(rawdata['Upfront_charges']) == True, 'status_upcharge'] = 1
    rawdata.loc[pd.isnull(rawdata['income']) == True, 'status_income'] = 1
    rawdata.loc[pd.isnull(rawdata['LTV']) == True, 'status_ltv'] = 1
    # set status to 1 if LTV is Null

    imp_fillmean = SimpleImputer(missing_values=np.nan, strategy='mean')
    rate_of_interest = rawdata.loc[:, 'rate_of_interest'].values.reshape(-1, 1)
    Upfront_charges = rawdata.loc[:, 'Upfront_charges'].values.reshape(-1, 1)
    income = rawdata.loc[:, 'income'].values.reshape(-1, 1)
    LTV = rawdata.loc[:, 'LTV'].values.reshape(-1, 1)
    rawdata['rate_of_interest'] = imp_fillmean.fit_transform(rate_of_interest)
    rawdata['Upfront_charges'] = imp_fillmean.fit_transform(Upfront_charges)
    rawdata['income'] = imp_fillmean.fit_transform(income)
    rawdata['LTV'] = imp_fillmean.fit_transform(LTV)
    # fill na with mean

    rawdata = pd.get_dummies(rawdata,
                             columns=['loan_limit', 'Gender', 'approv_in_adv', 'loan_purpose', 'Neg_ammortization',
                                      'total_units', 'age', 'Region'],
                             prefix=['loan_limit', 'Gender', 'approv_in_adv', 'loan_purpose', 'Neg_ammortization',
                                     'total_units', 'age', 'Region'])
    #transform into dummy

    target = rawdata['Status']
    data = rawdata.drop(['Status'], axis=1)
    return data, target


