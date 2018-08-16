from sklearn.preprocessing import RobustScaler
import pandas as pd
from imblearn.over_sampling import SMOTE
import time
import numpy as np


def load_oversampled_data(frac):
    data_frame = pd.read_csv('./datasets/oversampled_data.csv')
    data_frame = data_frame.sample(frac=frac)
    features = data_frame.drop('Class', axis=1)
    labels = data_frame['Class']
    print('data loaded.')
    return np.asarray(features.values, np.float32), np.asarray(labels.values, np.float32)


def load_original_data():
    data_frame = pd.read_csv('./datasets/data.csv')
    rob_scaler = RobustScaler()
    scaled_amount = rob_scaler.fit_transform(data_frame['Amount'].values.reshape(-1, 1))
    scaled_time = rob_scaler.fit_transform(data_frame['Time'].values.reshape(-1, 1))
    data_frame.drop(['Time', 'Amount'], axis=1, inplace=True)
    data_frame.insert(0, 'scaled_value', scaled_amount)
    data_frame.insert(1, 'scaled_time', scaled_time)
    features = data_frame.drop('Class', axis=1)
    labels = data_frame['Class']
    return np.asarray(features.values, np.float32), np.asarray(labels.values, np.float32)


if __name__ == '__main__':
    # load data
    df = pd.read_csv('./datasets/data.csv')
    # scale features
    rob_scaler = RobustScaler()
    scaled_amount = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    scaled_time = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    df.insert(0, 'scaled_value', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    # data divide(feature and label)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # data oversampling using somte
    t0 = time.time()
    sm = SMOTE(ratio='minority', random_state=42)
    Xsm_train, ysm_train = sm.fit_sample(X, y)
    ysm_train = np.reshape(ysm_train, (len(ysm_train), 1))
    t1 = time.time()
    print("SMOTE took {:.2} s".format(t1 - t0))

    oversampled_data = np.hstack((Xsm_train, ysm_train))
    print(Xsm_train.shape, ysm_train.shape, oversampled_data.shape)
    oversampled_df = pd.DataFrame(oversampled_data, columns=df.columns)
    oversampled_df = oversampled_df.sample(frac=1)
    print(oversampled_df)
    oversampled_df.to_csv('./datasets/oversampled_data.csv', index=None)
    # print(len(X),len(y))
    # print(len(Xsm_train),len(ysm_train))
