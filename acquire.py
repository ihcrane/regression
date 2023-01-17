from env import get_connection
import pandas as pd
import os

def get_titanic_data():
    
    '''
    This function is used to get titanic data from sql database.
    '''
    
    if os.path.isfile('titanic.csv'):
        
        return pd.read_csv('titanic.csv')
    
    else:
        
        url = get_connection('titanic_db')
        query = '''SELECT * FROM passengers'''
        df = pd.read_sql(query, url)
        df.to_csv('titanic.csv')
        return df

def get_iris_data(get_connection):
    
    '''
    This function is used to get iris data from sql database.
    '''
    
    if os.path.isfile('iris.csv'):
        
        return pd.read_csv('iris.csv')
    
    else:
        
        url = get_connection('iris_db')
        query = '''
                SELECT * FROM measurements 
                JOIN species USING(species_id)
                '''
        df = pd.read_sql(query, url)
        df.to_csv('iris.csv')
        return df

def get_telco_data(get_connection):
    
    '''
    This function is used to get titanic data from sql database.
    '''
    
    if os.path.isfile('telco.csv'):
        
        return pd.read_csv('telco.csv')
    
    else:
        
        url = get_connection('telco_churn')
        query = '''SELECT * FROM customers
                    JOIN internet_service_types USING(internet_service_type_id)
                    JOIN contract_types USING(contract_type_id)
                    JOIN payment_types USING(payment_type_id)
                    '''
        df = pd.read_sql(query, url)
        df.to_csv('telco.csv')
        return df

def get_zillow_data():
    
    '''
    This function is used to get zillow data from sql database.
    '''
    
    if os.path.isfile('zillow.csv'):
        
        return pd.read_csv('zillow.csv')
    
    else:
        
        url = get_connection('zillow')
        query = '''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
                FROM properties_2017
                WHERE propertylandusetypeid = 261;'''
        df = pd.read_sql(query, url)
        df.to_csv('zillow.csv')
        return df


def wrangle_zillow(df):
    
    '''
    This function is used to get zillow data from sql database, renaming columns, 
    dropping nulls and duplicates.
    '''
    
    df = get_zillow_data()
    
    # renaming columns
    df = df.rename(columns={'bedroomcnt':'bed',
                        'bathroomcnt':'bath',
                        'calculatedfinishedsquarefeet':'sqft',
                        'taxvaluedollarcnt':'tax_value',
                        'yearbuilt':'year'})
    
    # drop Unnamed: 0 column
    df = df.drop(columns=['Unnamed: 0'])

    #drop nulls
    df = df.dropna()
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def get_auto_mpg():
    
    '''Acquire, clean, and return the auto-mpg dataset'''
    
    df = pd.read_fwf('auto-mpg.data', header=None)
    
    df.columns = ['mpg', 'cylinders', 'displ', 'horsepower', 'weight', 'acc',
              'model_year', 'origin', 'name']
    
    df = df[df['horsepower'] != '?']
    
    df['horsepower'] = df['horsepower'].astype('float')
    
    return df