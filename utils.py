import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

def confusion_matrix(y_true,y_pred):
  confusion_matrix_ = np.sum(multilabel_confusion_matrix(y_true, y_pred),axis=0)
  recall = confusion_matrix_[1,1]/(confusion_matrix_[1,1]+confusion_matrix_[1,0])
  print("Confusion Matrix\n", confusion_matrix_)
  print(classification_report(y_true,y_pred))

def multi_cat_val_handler(df, cols_list):
  '''
  Turn single feature with multiple values into machine readable form
  cleaned = df['Hobby (Select 3)'].str.split(';', expand=True).stack()
  pd.get_dummies(cleaned, prefix='h').groupby(level=0).sum()
  '''

  for col in cols_list:
    prefix = col[0].lower() #first alphabet
    cleaned = df[col].str.split(';', expand=True).stack()
    temp_df = pd.get_dummies(cleaned, prefix=prefix).groupby(level=0).sum()
    df = df.join(temp_df)
    df.drop(col, axis=1, inplace=True)

  return df

def check_df(df, show_all=False, show_unique=False):
    null_exist =False
    for col in df.columns:
        if show_all or df[col].isna().values.sum() != 0:
            print(f'{col}: null={df[col].isna().values.sum()}, dtypes={df[col].dtypes}')
            if show_unique:
                print(f'{col}:', df[col].unique())
            if df[col].isna().values.sum() != 0:
                null_exist =True
    
    if not null_exist:
        print('No null exist in this dataframe.')


def data_preprocessing(df):
    df['Birthday'] = df['Birthday'].apply(lambda _: datetime.strptime(_,"%Y-%m-%d"))

    
    #create necessary columns
    df['Timestamp']= df['Timestamp'].astype("datetime64")
    df['Age'] = df['Timestamp'].dt.year - df['Birthday'].apply(lambda x: x.year)

    # Utilizing Relationship Period
    
    df['Relationship Period (Starting Date)'] = df['Relationship Period (Starting Date)'].astype("datetime64")
    df['Relationship Period'] = df['Timestamp'].dt.to_pydatetime() - df['Relationship Period (Starting Date)'].dt.to_pydatetime()
    df['Relationship Period'] = df['Relationship Period'].dt.days
    # Replacing Nan with 0 
    df['Relationship Period'] = df['Relationship Period'].replace(np.nan,0)
    
    #remove outliers
    df['Age'] = df['Age'].apply(lambda x: x if (x >18 and x <60) else df['Age'].median()) #use median to impute age
    
    
    #remove unnecessary columns
    df = df.drop(['Timestamp', 'Birthday', 'Relationship Period (Starting Date)'], axis=1)

    #rename for readability
    df.rename(columns={'Hobby (Select 3)': 'Hobby', 'Gift Preferences (Choose 3)': 'Gift Preferences'}, inplace=True)

    #Handle single feature with multiple values through one-hot encoding
    multi_val_cols = ['Hobby', 'Movie Preferences', 'Gift Preferences']
    new_df = multi_cat_val_handler(df, multi_val_cols)
    new_df = pd.get_dummies(new_df, drop_first=True)

    #remove weird col
    new_df.drop(['g_A smile HAHAHAHA'], axis=1, inplace=True)

    #remove NAN in movie - NAN means they don't like all movie 
    for col in new_df.columns:
      if col.startswith('m_'):
        new_df[col] = new_df[col].fillna(0)
        
    return new_df

def get_dataset():
    # returns data tuple with X_train, X_test, y_train, y_test
    df = pd.read_csv("data/Demographic to Gift Preference Survey.csv")
    new_df = data_preprocessing(df)
    y = new_df.iloc[:, 20:32]
    X = new_df.drop(y.columns, axis=1)
    # Using a MinMaxScaler
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
    