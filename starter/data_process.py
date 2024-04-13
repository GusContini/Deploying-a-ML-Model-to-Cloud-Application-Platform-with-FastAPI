''' Docstring
'''
from sklearn.model_selection import train_test_split
import pandas as pd


def split_data():
    ''' Docstring
    '''
    # read csv file into Pandas DataFrame
    df = pd.read_csv(
        '/home/chafund/GIT/Deploying-a-ML-Model-to-Cloud-Application'
        '-Platform-with-FastAPI/data/census.csv'
        )

    # Trim spaces on column's names
    cols = list(df.columns)
    cols = [col.strip() for col in cols]
    df.columns = cols

    # Split data into train and test sets
    train, test = train_test_split(df, test_size=0.20, random_state=42)

    train.to_csv('/home/chafund/GIT/Deploying-a-ML-Model-to-Cloud-Application'
                 '-Platform-with-FastAPI/data/train.csv')
    test.to_csv('/home/chafund/GIT/Deploying-a-ML-Model-to-Cloud-Application'
                '-Platform-with-FastAPI/data/test.csv')


if __name__ == '__main__':
    split_data()
