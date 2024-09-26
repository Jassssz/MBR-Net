import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

#load dataset
def load_dataset(file, sheet_name):
    df=pd.read_excel(file, sheet_name=sheet_name,skiprows=[1])

    df['CIP_Counter'] = df['CIP_Counter'].astype('float64')
    df['NaOCl'] = df['NaOCl'].astype('float64')
    df['Acid'] = df['Acid'].astype('float64')
    df['NaOCl_Amount'] = df['NaOCl_Amount'].astype('float64')
    df['Acid_Amount'] = df['Acid_Amount'].astype('float64')

    perm1_df=pd.DataFrame()
    perm1_df=pd.concat([df['Permeability'],
                        df['Flux'],
                        df['Feed_NH4N'],
                    ],axis=1)
    
    param2_df=pd.DataFrame()
    param2_df=pd.concat([df['Flux'],
                        df['Feed_NH4N'],
                        df['NaOCl'],
                        df['Acid'],
                        df['NaOCl_Amount'],
                        df['Acid_Amount'],
                        ], axis=1)
    time_df = df['Timestamp']

    return df, perm1_df, param2_df, time_df

#get parameters for clf
def perm_distribution(df):
    Perm_scaler=MinMaxScaler()
    perm_scaled=Perm_scaler.fit_transform(np.array(df['Permeability']).reshape(-1,1))

    Distribution = np.histogram(perm_scaled,bins=1000)
    means = np.mean(np.vstack([Distribution[1][:-1], Distribution[1][1:]]), axis=0)
    return means, Distribution

def polyfit(df):
    means, Distribution = perm_distribution(df)
    poly=np.polyfit(means[:340],Distribution[0][:340],9)
    p = np.poly1d(poly)
    p_list = list(np.float32(p.coeffs))
    # plt.plot(means[:330],Distribution[0][:330],'.',
    #         means[:330],p(means[:330]),'*', markersize=3)
    # plt.plot(means[330:],Distribution[0][330:])
    return p_list

def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c  
 
def expfit(df):
    means, Distribution = perm_distribution(df)
    x=means[350:]
    y=Distribution[0][350:]
    params, covariance = curve_fit(exponential_decay, x, y, bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))
    a_fit, b_fit, c_fit = params
    # plt.plot(means[330:], exponential_decay(means[330:], a_fit, b_fit, c_fit), '*', markersize=3, color='red', label='Fitted Curve')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Exponential Decay Fitting')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # print("Fitted parameters:")
    # print("a =", a_fit)
    # print("b =", b_fit)
    # print("c =", c_fit)
    # print(means[330])
    # print(exponential_decay(means[330],a_fit, b_fit, c_fit))
    # print(p(means[330]))
    return a_fit, b_fit, c_fit

#create subdatasets
def df_to_X_Y(df, perm_df, param_df, time_df):
    window_size = 900
    prediction_window_size = 180
    #split the datasetNum and TrainNum from df
    SetNum_max = int(df['SetNum'].max())
    TrainNum_max = int(df['Train'].max())
    X1_train, X1_valid, X1_test = [], [], []
    X2_train, X2_valid, X2_test = [], [], []
    Y_train, Y_valid, Y_test = [], [], []
    timestamp_test, timestamp_empty = [], []

    for set in range(1, SetNum_max+1):
        for Train in range(1, TrainNum_max+1):
            #X1 for testing, X6 and P4 for validation, others for training
            df_sliced_index = df[(df['SetNum']==set) & (df['Train']==Train)].index
            X1_df_sliced = perm_df.iloc[df_sliced_index].reset_index(drop=True).to_numpy()
            X2_df_sliced = param_df.iloc[df_sliced_index].reset_index(drop=True).to_numpy()
            time_df_sliced = time_df.iloc[df_sliced_index].reset_index(drop=True).to_numpy()
            
            X1_list, X2_list, Y_list, timestamp_list = X1_train, X2_train, Y_train, timestamp_empty
            if (Train == 1 and (df['Plant'][df_sliced_index] == 'X').all()):
                X1_list, X2_list, Y_list, timestamp_list = X1_test, X2_test, Y_test, timestamp_test
            elif (Train == 4 and (df['Plant'][df_sliced_index] == 'P').all()) or (Train == 6 and (df['Plant'][df_sliced_index] == 'X').all()) or (Train == 3 and (df['Plant'][df_sliced_index] == 'X').all()):
                X1_list, X2_list, Y_list = X1_valid, X2_valid, Y_valid

            for i in range(len(X1_df_sliced)-window_size-(prediction_window_size-1)):
                row_perm = X1_df_sliced[i:i+window_size]
                row_param = X2_df_sliced[i+window_size:i+window_size+prediction_window_size]
                label = X1_df_sliced[i+window_size:i+window_size+prediction_window_size][:,0]
                timestamp = time_df_sliced[i:i+window_size+prediction_window_size]

                X1_list.append(row_perm)
                X2_list.append(row_param)
                Y_list.append(label)
                timestamp_list.append(timestamp)

    return np.array(X1_train), np.array(X2_train), np.array(Y_train), np.array(X1_valid), np.array(X2_valid), np.array(Y_valid), np.array(X1_test), np.array(X2_test), np.array(Y_test), np.array(timestamp_test)

def shuffle_data(X1, X2, Y):
    randomize = np.arange(len(X1))
    np.random.shuffle(randomize)
    return X1[randomize], X2[randomize], Y[randomize]

def shuffle_apply_scaler(df, perm_df, param_df, time_df):
    X1_arr_train, X2_arr_train, Y_arr_train, X1_arr_valid, X2_arr_valid, Y_arr_valid, X1_arr_test, X2_arr_test, Y_arr_test, timestamp_test = df_to_X_Y(df, perm_df, param_df, time_df)
    
    #scaler for permeability
    scaler_perm = MinMaxScaler()
    scaler_perm.fit(np.array(df['Permeability']).reshape(-1,1))
    Y_arr_train_scaled = scaler_perm.transform(Y_arr_train.reshape(-1,1)).reshape(-1,180)
    Y_arr_valid_scaled = scaler_perm.transform(Y_arr_valid.reshape(-1,1)).reshape(-1,180)
    Y_arr_test_scaled = scaler_perm.transform(Y_arr_test.reshape(-1,1)).reshape(-1,180)

    X1_perm_train_scaled = scaler_perm.transform(X1_arr_train[:,:,0].reshape(-1,1))
    X1_perm_valid_scaled = scaler_perm.transform(X1_arr_valid[:,:,0].reshape(-1,1))
    X1_perm_test_scaled = scaler_perm.transform(X1_arr_test[:,:,0].reshape(-1,1))

    #scaler for flux and NH4N
    scaler_excPerm = MinMaxScaler()
    scaler_excPerm.fit(np.array([df['Flux'],df['Feed_NH4N']]).T)
    X1_excPerm_train_scaled = scaler_excPerm.transform(X1_arr_train[:,:,1:].reshape(-1,2))
    X1_excPerm_valid_scaled = scaler_excPerm.transform(X1_arr_valid[:,:,1:].reshape(-1,2))
    X1_excPerm_test_scaled = scaler_excPerm.transform(X1_arr_test[:,:,1:].reshape(-1,2))
    #combine and get the scaled X1
    X1_arr_train_scaled = np.append(X1_perm_train_scaled,X1_excPerm_train_scaled,axis=1).reshape(X1_arr_train.shape[0], X1_arr_train.shape[1], X1_arr_train.shape[2])
    X1_arr_valid_scaled = np.append(X1_perm_valid_scaled,X1_excPerm_valid_scaled,axis=1).reshape(X1_arr_valid.shape[0], X1_arr_valid.shape[1], X1_arr_valid.shape[2])
    X1_arr_test_scaled = np.append(X1_perm_test_scaled,X1_excPerm_test_scaled,axis=1).reshape(X1_arr_test.shape[0], X1_arr_test.shape[1], X1_arr_test.shape[2])   

    #scaler for X2
    scaler_Param = MinMaxScaler()
    scaler_Param.fit(np.array(param_df))
    X2_arr_train_scaled = scaler_Param.transform(X2_arr_train.reshape(-1,param_df.shape[1])).reshape(X2_arr_train.shape[0], X2_arr_train.shape[1], X2_arr_train.shape[2])
    X2_arr_valid_scaled = scaler_Param.transform(X2_arr_valid.reshape(-1,param_df.shape[1])).reshape(X2_arr_valid.shape[0], X2_arr_valid.shape[1], X2_arr_valid.shape[2])
    X2_arr_test_scaled = scaler_Param.transform(X2_arr_test.reshape(-1,param_df.shape[1])).reshape(X2_arr_test.shape[0], X2_arr_test.shape[1], X2_arr_test.shape[2])

    ###
    #Randomly shuffle the first axis of the 3darray
    X1_arr_train_shuffle, X2_arr_train_shuffle, Y_arr_train_shuffle = shuffle_data(X1_arr_train_scaled, X2_arr_train_scaled, Y_arr_train_scaled)
    X1_arr_valid_shuffle, X2_arr_valid_shuffle, Y_arr_valid_shuffle = shuffle_data(X1_arr_valid_scaled, X2_arr_valid_scaled, Y_arr_valid_scaled)
    # X1_arr_test_shuffle, X2_arr_test_shuffle, Y_arr_test_shuffle = shuffle_data(X1_arr_test_scaled, X2_arr_test_scaled, Y_arr_test_scaled)

    
    # return X1_arr_train_shuffle, X2_arr_train_shuffle, Y_arr_train_shuffle, X1_arr_valid_shuffle, X2_arr_valid_shuffle, Y_arr_valid_shuffle, X1_arr_test_shuffle, X2_arr_test_shuffle, Y_arr_test_shuffle
    return X1_arr_train_shuffle, X2_arr_train_shuffle, Y_arr_train_shuffle, X1_arr_valid_shuffle, X2_arr_valid_shuffle, Y_arr_valid_shuffle, X1_arr_test_scaled, X2_arr_test_scaled, Y_arr_test_scaled, timestamp_test


# create and transform subdatasets for testing
def transform(df, testing_df):
    max_value_series = df.max()
    min_value_series = df.min()
    for i in range(len(testing_df.columns)):
        max_value = max_value_series[testing_df.columns[i]]
        min_value = min_value_series[testing_df.columns[i]]
        testing_df[testing_df.columns[i]] = (testing_df[testing_df.columns[i]]-min_value)/(max_value-min_value)
    return testing_df

def df_to_X_Y_testing(df, perm_df, param_df):
    #split the datasetNum and TrainNum from df
    SetNum_max = int(df['SetNum'].max())
    TrainNum_max = int(df['Train'].max())
    X1, X2, Y = [], [], []
    window_size = 900
    prediction_window_size = 180

    for set in range(1, SetNum_max+1):
        for Train in range(1, TrainNum_max+1):
            #X1 for testing, X6 and P4 for validation, others for training
            df_sliced_index = df[(df['SetNum']==set) & (df['Train']==Train)].index
            X1_df_sliced = perm_df.iloc[df_sliced_index].reset_index(drop=True).to_numpy()
            X2_df_sliced = param_df.iloc[df_sliced_index].reset_index(drop=True).to_numpy()
            

            for i in range(len(X1_df_sliced)-window_size-(prediction_window_size-1)):
                row_perm = X1_df_sliced[i:i+window_size]
                row_param = X2_df_sliced[i+window_size:i+window_size+prediction_window_size]
                label = X1_df_sliced[i+window_size:i+window_size+prediction_window_size][:,0]
                X1.append(row_perm)
                X2.append(row_param)
                Y.append(label)

    return np.array(X1), np.array(X2), np.array(Y)

def shuffle_data_testing(X1, X2, Y):
    randomize = np.arange(len(X1))
    np.random.shuffle(randomize)
    return X1[randomize], X2[randomize], Y[randomize]

def shuffle_apply_scaler_testing(df,df_testing,perm_df_testing,param_df_testing):
    perm_df_testing_scaled = transform(df, perm_df_testing)
    param_df_testing_scaled = transform(df, param_df_testing)
    
    X1_arr, X2_arr, Y_arr = df_to_X_Y_testing(df_testing, perm_df_testing_scaled, param_df_testing_scaled)
    ###
    #Randomly shuffle the first axis of the 3darray
    X1_arr_shuffle, X2_arr_shuffle, Y_arr_shuffle = shuffle_data_testing(X1_arr, X2_arr, Y_arr)

    return X1_arr_shuffle, X2_arr_shuffle, Y_arr_shuffle


