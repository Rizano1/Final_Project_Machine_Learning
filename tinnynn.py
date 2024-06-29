import argparse
import os
import numpy as np
import tinynn as tn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from scipy.signal import butter, sosfilt

def band_pass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    filtered_data = sosfilt(sos, data, axis=1)
    return filtered_data


def load_data():
    URL_ = './PPG_Dataset/Labels/Total.csv'
    data = pd.read_csv(URL_)
    data['Gender'] = np.where(data['Gender'] == 'Female', 0, 1)
    file_path = './PPG_Dataset/RawData/'

    all_ppg_data = pd.DataFrame()
    
    for file_name in os.listdir(file_path):
        if file_name.endswith('.csv'):
            file = os.path.join(file_path, file_name)
            ppg_data = pd.read_csv(file, header=None)            
            # Apply band-pass filter
            ppg_data = ppg_data.T
            ppg_data = ppg_data.values
            filtered_ppg_data = np.hstack([
                band_pass_filter(ppg_data, 800, 900, fs=21901),
                band_pass_filter(ppg_data, 900, 1000, fs=21901),
                band_pass_filter(ppg_data, 1100, 1200, fs=21901)
            ])
            ppg_data = pd.DataFrame(filtered_ppg_data)

            all_ppg_data = pd.concat([all_ppg_data, ppg_data], ignore_index=True)
    
    data = pd.concat([data, all_ppg_data], axis=1)
    data.drop('ID', axis=1, inplace=True)
    cols = data.columns.tolist()
    cols = cols[:2] + cols[3:] + [cols[2]]
    data = data[cols]

    # Convert all column names to strings
    data.columns = data.columns.astype(str)
    # Normalizing the data

    # scaler = MinMaxScaler()
    # data[data.columns[4:-1]] = scaler.fit_transform(data[data.columns[4:-1:]])
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)
    train_data = np.asarray(train_data, dtype='float32')
    test_data = np.asarray(test_data, dtype='float32')
    return train_data, test_data

def main():
    train_data, test_data = load_data()

    x_train = train_data[:, :-1]
    y_train = train_data[:, -1].reshape(-1, 1)

    net = tn.net.Net([
        tn.layer.Dense(8),
        tn.layer.ReLU(),
        tn.layer.Dropout(1),  # Dropout layer with 50% dropout rate
        tn.layer.Dense(8),
        tn.layer.ReLU(),
        tn.layer.Dropout(1),  # Dropout layer with 50% dropout rate
        tn.layer.Dense(8),
        tn.layer.ReLU(),
        tn.layer.Dropout(1),  # Dropout layer with 50% dropout rate  # Dropout layer with 50% dropout rate
        tn.layer.Dense(1)
    ])

    model = tn.model.Model(net=net, loss=tn.loss.MSE(), optimizer=tn.optimizer.Adam(weight_decay=0.001))  # L2 regularization
    iterator = tn.data_iterator.BatchIterator(batch_size=32)

    for epoch in range(30):
      for batch in iterator(x_train, y_train):
          preds = model.forward(batch.inputs)
          _, grads = model.backward(preds, batch.targets)
          model.apply_grads(grads)

      # Evaluate on training data
      preds = net.forward(x_train)
      mse, info = tn.metric.mean_square_error(preds, y_train)
      print(f"Epoch {epoch} mse: {mse} info: {info}")

    # Save the model
    model.save('model.pkl')
    # Evaluate on test data
    x_test = test_data[:, :-1]
    y_test = test_data[:, -1].reshape(-1, 1)

    preds = net.forward(x_test)
    test_mse, test_info = tn.metric.mean_square_error(preds, y_test)
    print(f"Test mse: {test_mse} info: {test_info}")

def evaluate(model, test_x, test_y):
    model.is_training = False
    test_pred_score = model.forward(test_x)
    print(test_pred_score)

if __name__ == '__main__':
    main()
