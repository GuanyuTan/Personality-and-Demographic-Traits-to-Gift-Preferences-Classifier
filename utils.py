import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate

def confusion_matrix(y_true,y_pred):
    '''
    Return a classifaction report to visualize the scores.
    '''

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
    '''
    Check whether null value exist in the dataframe.
    '''
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
    '''
    Clean up the data.
    '''

    df['Birthday'] = df['Birthday'].apply(lambda _: datetime.strptime(_,"%Y-%m-%d"))
    
    #create necessary columns
    df['Timestamp']= df['Timestamp'].astype("datetime64")
    df['Age'] = df['Timestamp'].dt.year - df['Birthday'].apply(lambda x: x.year)

    # Utilizing Relationship Period
    # Will not be using Relationship Period due to low correlation with output     
#     df['Relationship Period (Starting Date)'] = df['Relationship Period (Starting Date)'].astype("datetime64")
#     df['Relationship Period'] = df['Timestamp'].dt.to_pydatetime() - df['Relationship Period (Starting Date)'].dt.to_pydatetime()
#     df['Relationship Period'] = df['Relationship Period'].dt.days
#     # Replacing Nan with 0 
#     df['Relationship Period'] = df['Relationship Period'].replace(np.nan,0)
    
    #remove outliers
    df['Age'] = df['Age'].apply(lambda x: x if (x >18 and x <60) else df['Age'].median()) #use median to impute age
    
    #remove unnecessary columns
    df = df.drop(['Timestamp', 'Birthday', 'Relationship Period (Starting Date)'], axis=1)

    #rename for readability
    df.rename(columns={'Hobby (Select 3)': 'Hobby', 'Gift Preferences (Choose 3)': 'Gift Preferences'}, inplace=True)

    #Handle single feature with multiple values through one-hot encoding
    multi_val_cols = ['Hobby', 'Movie Preferences', 'Gift Preferences']
    
    #Body State removed due to imbalanced variable
    df.drop(["Body State"], axis=1, inplace=True)
    new_df = multi_cat_val_handler(df, multi_val_cols)
    new_df = pd.get_dummies(new_df, drop_first=True)
    
    #remove weird col
    new_df.drop(['g_A smile HAHAHAHA'], axis=1, inplace=True)
    new_df.drop(['g_I would choose all'], axis=1, inplace=True)
    new_df.drop(['g_Not necessarily'], axis=1, inplace=True)
    
    # Through the correlation matrix, we try to eliminate the variables with lesser correlation values (value<|0.15|).
    # h_Collecting eliminated due to low frequency.
    # We eliminate Body_State, Relationship Status due to the highly imbalanced data. 
    new_df.drop(["Relationship Status_Married"], axis=1, inplace=True)
    # We eliminate Relationship Period, m_Drama, How_do_you_get_your_energy_I due to low correlation values.
    new_df.drop(["h_Collecting (stamps, coins)"], axis=1, inplace=True)
    new_df.drop(["m_Drama","How do you get your energy?_I"],axis=1, inplace=True)
    #remove NAN in movie - NAN means they don't like all movie 
    for col in new_df.columns:
        if col.startswith('m_'):
            new_df[col] = new_df[col].fillna(0)
        
    return new_df

def get_dataset(): 
    '''
    Get the datasets.
    '''
    # returns data tuple with X_train, X_test, y_train, y_test
    df = pd.read_csv("data/Demographic to Gift Preference Survey.csv")
    new_df = data_preprocessing(df)
    mask = new_df.columns.str.contains(r'g_.*')
    new_df = new_df[new_df.iloc[:, mask].sum(axis=1) ==3] #remove those who don't have 3 options
    y = new_df.iloc[:, mask]
    X = new_df.drop(y.columns, axis=1)
    # Using a MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)
    return X, y, X_train, X_test, y_train, y_test

def cross_validate_models(models, X, y):
    '''
    cross_validate and return the best model.
    '''
    features = ['Models',
                'test_f1_macro_mean', 'test_f1_macro_std', 'train_f1_macro_mean', 'train_f1_macro_std',
                'test_f1_weighted_mean', 'test_f1_weighted_std', 'train_f1_weighted_mean', 'train_f1_weighted_std',
                'test_f1_micro_mean', 'test_f1_micro_std', 'train_f1_micro_mean', 'train_f1_micro_std',
                'test_recall_weighted_mean', 'test_recall_weighted_std', 'train_recall_weighted_mean', 'train_recall_weighted_std',
                'test_recall_macro_mean', 'test_recall_macro_std', 'train_recall_macro_mean', 'train_recall_macro_std',
                'test_recall_micro_mean', 'test_recall_micro_std', 'train_recall_micro_mean', 'train_recall_micro_std',
                'test_precision_weighted_mean', 'test_precision_weighted_std', 'train_precision_weighted_mean', 'train_precision_weighted_std', 
                'test_precision_macro_mean', 'test_precision_macro_std', 'train_precision_macro_mean', 'train_precision_macro_std', 
                'test_precision_micro_mean', 'test_precision_micro_std', 'train_precision_micro_mean', 'train_precision_micro_std', 
                ]
                
    scoring =['f1_weighted', 'f1_macro', 'f1_micro', 
              'recall_weighted', 'recall_macro', 'recall_micro',
              'precision_weighted', 'precision_macro', 'precision_micro']
    
    df = pd.DataFrame(columns = features) 

    best_f1_macro_mean = 0
    best_model = None

    for index, model in enumerate(models):
        #drop all columns that has NAN
#             X = df_32_X.drop(col, axis=1)
        rkf = RepeatedKFold(n_splits=5, n_repeats=15, random_state=42) #since dataset is low, cv=~5 (5 folds) and repeat 15 times
        cv = cross_validate(model, X, y, cv=rkf, return_train_score=True, scoring=scoring)

        info = pd.Series({'Models':model, 
                          
                          'test_f1_macro_mean':cv['test_f1_macro'].mean(), 'test_f1_macro_std':cv['test_f1_macro'].std(), 
                          'train_f1_macro_mean':cv['train_f1_macro'].mean(), 'train_f1_macro_std':cv['train_f1_macro'].std(),

                          'test_f1_weighted_mean':cv['test_f1_weighted'].mean(), 'test_f1_weighted_std':cv['test_f1_weighted'].std(), 
                          'train_f1_weighted_mean':cv['train_f1_weighted'].mean(), 'train_f1_weighted_std':cv['train_f1_weighted'].std(),

                          'test_f1_micro_mean':cv['test_f1_micro'].mean(), 'test_f1_micro_std':cv['test_f1_micro'].std(), 
                          'train_f1_micro_mean':cv['train_f1_micro'].mean(), 'train_f1_micro_std':cv['train_f1_micro'].std(),

                          'test_recall_weighted_mean':cv['test_recall_weighted'].mean(), 'test_recall_weighted_std':cv['test_recall_weighted'].std(), 
                          'train_recall_weighted_mean':cv['train_recall_weighted'].mean(), 'train_recall_weighted_std':cv['train_recall_weighted'].std(),

                          'test_recall_macro_mean':cv['test_recall_macro'].mean(), 'test_recall_macro_std':cv['test_recall_macro'].std(), 
                          'train_recall_macro_mean':cv['train_recall_macro'].mean(), 'train_recall_macro_std':cv['train_recall_macro'].std(),

                          'test_recall_micro_mean':cv['test_recall_micro'].mean(), 'test_recall_micro_std':cv['test_recall_micro'].std(), 
                          'train_recall_micro_mean':cv['train_recall_micro'].mean(), 'train_recall_micro_std':cv['train_recall_micro'].std(),

                          'test_precision_weighted_mean':cv['test_precision_weighted'].mean(), 'test_precision_weighted_std':cv['test_precision_weighted'].std(), 
                          'train_precision_weighted_mean':cv['train_precision_weighted'].mean(), 'train_precision_weighted_std':cv['train_precision_weighted'].std(),

                          'test_precision_macro_mean':cv['test_precision_macro'].mean(), 'test_precision_macro_std':cv['test_precision_macro'].std(), 
                          'train_precision_macro_mean':cv['train_precision_macro'].mean(), 'train_precision_macro_std':cv['train_precision_macro'].std(),

                          'test_precision_micro_mean':cv['test_precision_micro'].mean(), 'test_precision_micro_std':cv['test_precision_micro'].std(), 
                          'train_precision_micro_mean':cv['train_precision_micro'].mean(), 'train_precision_micro_std':cv['train_precision_micro'].std(),                          

                         })
                     
        df.loc[index] = info

        if best_f1_macro_mean < cv['test_f1_macro'].mean() and cv['train_f1_macro'].mean() >= 0.8:
            best_f1_macro_mean = cv['test_f1_macro'].mean()
            best_model = model

    display(df)
    print(f'The best model is \n {best_model} \n with f1_macro score: {best_f1_macro_mean}')
    return df, best_model, best_f1_macro_mean


## Deep Learning

import tensorflow as tf
from keras.losses import binary_crossentropy, categorical_crossentropy
import keras.backend as K
import numpy as np
from prettytable import PrettyTable
from prettytable import ALL
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def declare_model(input_shape, hidden_size, output_shape):
    '''
    Declare the models
    '''
    return nn.Sequential(nn.Linear(input_shape, hidden_size),
                        nn.Dropout(p=0.7),
                        nn.Sigmoid(),
                        # nn.Linear(64, 32),
                        # nn.Dropout(p=0.4),
                        nn.Linear(hidden_size, output_shape)
                        # nn.ReLU(),
                        # nn.Linear(32, 16),
                        # nn.Dropout(p=0.4),
                        # nn.ReLU(),                                 
                        # nn.Linear(16, y_test.shape[1])
                        )

def get_top_k_pred(y_pred, k_num):
    '''
    Get the top k predictions
    '''
    ans_master = []
    topk_master_list =[]
    for j in y_pred:#for each row:
        temp_ans =[]
        top_k_val_list = torch.topk(j, k_num).indices.numpy()
        topk_master_list.append(top_k_val_list)
        for index, k in enumerate(j):#for each prediction:
            if index in top_k_val_list: #if its top 3
                temp_ans.append(1)
            else:
                temp_ans.append(0)
        ans_master.append(temp_ans)
    return ans_master, topk_master_list

def plot_graph(epoch_length, epoch_step, train_loss_per_epoch_list, test_loss_per_epoch_list):
    x = range(0, epoch_length, epoch_step)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x, train_loss_per_epoch_list, label='Train')
    plt.plot(x, test_loss_per_epoch_list, label='Test')
    plt.legend()

def train_model(train_dataset, test_dataset, model, loss_fn, opt='Adam' , epochs=5000, k_num=3):
    '''
    Train the model
    '''
    trainloader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=128)
    testloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=128)

    criterion = loss_fn
    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    elif opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01)


#   this running_loss will keep track of the losses of every epoch from each respective iteration
    running_loss = 0.0
    accuracy = 0
    train_loss_per_epoch_list = []
    test_loss_per_epoch_list = []
    for epoch in range(1, epochs + 1):
        train_loss_per_epoch = 0
        model.train()
        train_acc_list = []
        train_top_confi_acc =[]

        for i, (features,labels) in enumerate(trainloader):
#           zero the parameter gradients
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss_per_epoch += loss.item()
            _, pred_index = get_top_k_pred(predictions, k_num)
            _, label_index = get_top_k_pred(labels, k_num)
            _, top_confi_index = get_top_k_pred(predictions, 1)

            for pred, lab, top_con in zip(pred_index, label_index, top_confi_index):
                correct_num = 0
                for p in pred:
                    if p in lab:
                        correct_num +=1
                train_acc_list.append(correct_num/len(lab))

                #top
                if top_con in lab:
                    train_top_confi_acc.append(1)
                else:
                    train_top_confi_acc.append(0)

        train_accuracy = np.array(train_acc_list).mean()
        train_top_confi_acc = np.array(train_top_confi_acc).mean()

        test_loss_per_epoch = 0
        test_acc_list =[]
        test_top_confi_acc =[]

        model.eval()
        for features, labels in testloader:
            with torch.no_grad():
                predictions = model(features)
                loss = criterion(predictions, labels)
                test_loss_per_epoch += loss.item()
                _, pred_index = get_top_k_pred(predictions, k_num)
                _, label_index = get_top_k_pred(labels, k_num)
                _, top_confi_index = get_top_k_pred(predictions, 1)

                for pred, lab, top_con in zip(pred_index, label_index, top_confi_index):
                    correct_num = 0
                    for p in pred:
                        if p in lab:
                            correct_num +=1
                    test_acc_list.append(correct_num/len(lab))

                    #top
                    if top_con in lab:
                        test_top_confi_acc.append(1)
                    else:
                        test_top_confi_acc.append(0)
                    
                    # print('test_top_con:', top_con)
                    # print('test_lab:', lab)

        test_accuracy = np.array(test_acc_list).mean()
        test_top_confi_acc = np.array(test_top_confi_acc).mean()

        if epoch%100 ==0:
            # print('test_top_con:', top_con)
            # print('test_lab:', lab)
            train_loss_per_epoch_list.append(train_loss_per_epoch)
            test_loss_per_epoch_list.append(test_loss_per_epoch)
            print(f"Epoch {epoch}, train_loss: {train_loss_per_epoch:.04f}, train_acc:{train_accuracy:.04f}, test_loss:{test_loss_per_epoch:.04f}, test_acc:{test_accuracy:.04f}, train_top_acc:{train_top_confi_acc:.04f}, test_top_acc:{test_top_confi_acc:.04f}") 

    plot_graph(epochs, 100, train_loss_per_epoch_list, test_loss_per_epoch_list)

def acc_topk(preds, targets, k=3):
    with torch.no_grad():
        topk = 0
        total = 0
        _, maxk = torch.topk(preds, k, dim=-1)
        maxk,_= torch.sort(maxk)
        for i,j in enumerate(targets):
            row = torch.tensor(np.where(j==1))
            for r in row[0]:
                total+=1
                if r in maxk[i]:
                    topk+=1
    return topk,total

def get_confusion_matrix_alvin(test_loader, train_loader, model):
    '''
    Predictions will always be 3 '1's.
    '''
    print('-------------------------------Test------------------------')
    for feature,label in test_loader:
        with torch.no_grad():
            predictions = model(feature)
            _,pred = torch.topk(predictions, 3, dim=-1)
            p = torch.zeros_like(label)
            for a,i in enumerate(pred):
                for k in i:
                    p[a][k] = 1
            top3,total = acc_topk(predictions, label, 3)
            accuracy = top3/(total)
            confusion_matrix_ = np.sum(multilabel_confusion_matrix(label, p),axis=0)
            recall = confusion_matrix_[1,1]/(confusion_matrix_[1,1]+confusion_matrix_[1,0])
            print("Confusion Matrix\n", confusion_matrix_)
            print(classification_report(label,p))

    print('-------------------------------Test------------------------')

    print('------------------------------Train------------------------')
    for feature,label in train_loader:
        with torch.no_grad():
            predictions = model(feature)
            _,pred = torch.topk(predictions, 3, dim=-1)
            p = torch.zeros_like(label)
            for a,i in enumerate(pred):
                for k in i:
                    p[a][k] = 1
            top3,total = acc_topk(predictions, label, 3)
            accuracy = top3/(total)
            confusion_matrix_ = np.sum(multilabel_confusion_matrix(label, p),axis=0)
            recall = confusion_matrix_[1,1]/(confusion_matrix_[1,1]+confusion_matrix_[1,0])
            print("Confusion Matrix\n", confusion_matrix_)
            print(classification_report(label,p))

    print('------------------------------Train------------------------')

def get_confusion_matrix_yc(test_loader, train_loader, model, threshold=0):
    '''
    Predictions '1's is based on the threshold.
    '''
    print('-------------------------------Test------------------------')
    for feature,label in test_loader:
        with torch.no_grad():
            y_pred = model(feature)
            y_pred = np.where(y_pred >threshold, 1, 0)
            confusion_matrix(label, y_pred)
    print('-------------------------------Test------------------------')

    print('------------------------------Train------------------------')
    for feature,label in train_loader:
        with torch.no_grad():
            y_pred = model(feature)
            y_pred = np.where(y_pred >threshold, 1, 0)
            confusion_matrix(label, y_pred)

    print('------------------------------Train------------------------')











