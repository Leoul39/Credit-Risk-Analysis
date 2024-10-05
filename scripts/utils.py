import pandas as pd
def features(df):
    """
    This function performs a feature engineering task by making new features from existing features.

    Parameter:- an original dataframe

    Returns:- a new dataframe that contains new features to enhance the predictive accuracy
    """
    # Changing the time columns to datetime data type
    df['TransactionStartTime']=pd.to_datetime(df['TransactionStartTime'])
    # For hours
    df['Hour']=df['TransactionStartTime'].dt.hour
    # For days
    df['Day']=df['TransactionStartTime'].dt.day
    # For months
    df['Month']=df['TransactionStartTime'].dt.month
    # For years
    df['Year']=df['TransactionStartTime'].dt.year
    # Sum of all transaction amounts for each customer.
    df['TotalTransactionAmount']=df.groupby('AccountId')['Amount'].transform('sum')
    # Average transaction amount per customer.
    df['AverageTransactionAmount']=df.groupby('AccountId')['Amount'].transform('mean')
    # Number of transactions per customer.
    df['TransactionCount']=df.groupby('AccountId')['Amount'].transform('count')
    # Variability of transaction amounts per customer.
    df['TransactionStd']=df.groupby('AccountId')['Amount'].transform('std')
    # Filling null values that can be made because of the standard deviation
    df.fillna(0,inplace=True)
    return df