import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import csv
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from sklearn.model_selection import StratifiedKFold


class Data():
    def __init__(self, data_type='train', max_files=None, normalize=False):
        self.normalize_factors = self.__get_persisted_normalize_factors()
        self.api_size_list = [8, 4, 16, 16, 8, 12, 16, 12] + [1] * 10
        
        if data_type == 'train':
            num_of_files = 18662
            y_map = pd.read_csv(os.getcwd()+'/dataset/train_kaggle.csv', sep=',', usecols=["Label"])
        if data_type == 'test':
            num_of_files = 6051
            if not self.normalize_factors:
                raise AttributeError("No normalize factor found to normalize test data. Please load training data at least once to generate normalize factors.")

        if (max_files != None):
            num_of_files = min(max_files, num_of_files)

        max_x = 1000
        x_width = 102

        self.x = np.zeros((num_of_files, max_x, x_width))
        self.y = np.zeros((num_of_files, 1))
        
        with tqdm(total=num_of_files, bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
            for i in range(num_of_files):
                x = np.load(os.getcwd()+'/dataset/' + data_type +
                            '/' + data_type+'/' + str(i) + '.npy')
                self.x[i, 0:0 + x.shape[0]] = x
                if data_type == 'train':
                    self.y[i, 0] = y_map.values[i][0]
                pbar.update(1)
                pbar.set_description(
                    'Loading ' + data_type + ' data' + ' %g' % (i+1) + '/'+str(num_of_files))

        # Normalize input features
        if (normalize):
            self.__normalize_x()
        
    def __get_persisted_normalize_factors(self):
        try:
            normalize_factors = []
            with open(os.getcwd()+'/dataset/normalize_factor.csv', 'rt') as normalize_factor_file:
                csv_reader = csv.reader(normalize_factor_file)
                for row in csv_reader:
                    normalize_factors.append(float(row[0]))
                return normalize_factors
        except FileNotFoundError:
            return []
        
    def __generate_normalize_factors(self):
        current_pos = 0
        for api_size in self.api_size_list:
            normalize_factor = self.__get_normalize_factor(self.x[:, :, current_pos:current_pos+api_size])
            self.normalize_factors.append(normalize_factor)
            current_pos += api_size
            
    def __get_normalize_factor(self, feature):
        abs_max_value = np.max(abs(feature))
        if not np.isscalar(abs_max_value):
            abs_max_value = np.max(abs_max_value)
        return abs_max_value
            
    def __persist_normalize_factors(self):
        with open(os.getcwd()+'/dataset/normalize_factor.csv', 'wt', newline='', encoding='utf-8') as normalize_factor_file:
            csv_writer = csv.writer(normalize_factor_file)
            csv_writer.writerows(map(lambda x:[x], self.normalize_factors))
    
    def __normalize_x(self):       
        if not self.normalize_factors:
            print("No existing normalize factors found. Generating new ones and persisting.")
            self.__generate_normalize_factors()
            self.__persist_normalize_factors()
        
        print("Normalizing x data using normalize factors: ", self.normalize_factors)
        current_pos = 0
        for index, api_size in enumerate(self.api_size_list):
            self.x[:, :, current_pos:current_pos+api_size] = self.x[:, :, current_pos:current_pos+api_size] / self.normalize_factors[index]
            current_pos += api_size
        

def create_model_a(show_summary = False):
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True),input_shape=(1000,102)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
])

    model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4,0.9,0.999),
              metrics=['accuracy',tf.keras.metrics.AUC()])
    
    if show_summary:
        model.summary()
    
    return model

def train_model_a_boosting():
    model_name ='model_a'
    full_train = Data(data_type='train')
    full_train_x,full_train_y = full_train.x, full_train.y
    del full_train

    val_data_x, train_data_x = full_train_x[:full_train_x.shape[0]//6,:], full_train_x[full_train_x.shape[0]//6:,:]
    val_data_y,train_data_y = full_train_y[:full_train_y.shape[0]//6,:], full_train_y[full_train_y.shape[0]//6:,:]

    print("Train data: x:"+ str(train_data_x.shape)+ ' y:'+str(train_data_y.shape))
    print("Validation data: x:"+ str(val_data_x.shape)+ ' y:'+str(val_data_y.shape))

    training_loss = np.ones(train_data_x.shape[0])
    for bag in range(10):
        print("Training bag"+str(bag))
        # Put data into bags
        model = create_model_a(False)
        train_size = train_data_x.shape[0]//10*6
        if bag != 0 :
            # Check prediction for training data
            for validate_bag in range(bag):
                checkpoint_dir = './checkpoints/' + model_name+ '_bag'+ str(validate_bag)
                model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
                print('validating bag '+str(validate_bag))
                if validate_bag == 0:
                    predictions = model.predict(train_data_x)
                else:
                    predictions = np.concatenate((predictions,model.predict(train_data_x)),axis=1)

            avg_predictions = np.mean(predictions, axis=1)
            
            # Select training data poorly predicted 
            training_loss = avg_predictions-np.reshape(train_data_y,(train_data_y.shape[0]))
            training_loss = np.absolute(training_loss)
        
        weights = training_loss / np.sum(training_loss, axis=0)
        idx = np.random.choice(np.arange(0, weights.shape[0]),train_size, p=weights)
        bag_x,bag_y = train_data_x[idx], train_data_y[idx]
            
        del model
            
        model = create_model_a(bag==0)
        # Directory where the checkpoints will be saved
        checkpoint_dir = './checkpoints/' + model_name+ '_bag'+ str(bag)
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=0, 
            patience=40, 
            mode='auto', 
            restore_best_weights=True)
            
        model.fit(x=bag_x,y=bag_y, validation_data=(val_data_x,val_data_y), epochs=250,batch_size=16,
                        callbacks=[checkpoint_callback,early_stopping_callback], verbose=1,shuffle=True)

def predict_model_a():
    model_name ='model_a'
    checkpoint_dir = './checkpoints/' + model_name
    model = create_model_a()
    test_data = Data(data_type='test')
    print("Test data: x:"+ str(test_data.x.shape)+ ' y:'+str(test_data.y.shape))

    for bag in range(10):
        print('predict bag '+str(bag))
        checkpoint_dir = './checkpoints/' + model_name+ '_bag'+ str(bag)
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        if bag == 0:
            predictions = model.predict(test_data.x,verbose=2)
        else:
            predictions = np.concatenate((predictions,model.predict(test_data.x,verbose=2)),axis=1)
    print(predictions.shape)

    avg_predictions = np.mean(predictions, axis=1)

    print(avg_predictions.shape)

    output_file = 'result_' + model_name + '.csv'
    with open(output_file, 'wt', newline='', encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(('Id', 'Predicted'))
        for id, predict in enumerate(avg_predictions):
            csv_writer.writerow((id, predict))
    print('Saved to '+ model_name + '.csv')

def train_model_b_cv():
    seed = 5242
    np.random.seed(seed)
    splits = 10

    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    full_train = Data(data_type='train')
    fold = 0

    for train, test in kfold.split(full_train.x, full_train.y):
        print('fold '+str(fold))
        model_name = 'model_b'
        
        # create model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True),input_shape=(1000,102)))
        model.add(tf.keras.layers.GlobalMaxPooling1D())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=0.2))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        checkpoint_dir = './cross_validation/' + model_name+ '_fold'+ str(fold)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=0, 
            patience=20, 
            mode='auto', 
            restore_best_weights=True)
    
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-4,0.9,0.999), 
                    metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
        # Train model
        model.fit(full_train.x[train], full_train.y[train], validation_data=(full_train.x[test],full_train.y[test]), 
                epochs=200, batch_size=32, verbose=1,callbacks=[checkpoint_callback,early_stopping_callback],shuffle=True)
        
        fold += 1

def predict_model_b():
    model_name ='model_b'
    splits = 10
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True),input_shape=(1000,102)))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


    test_data = Data(data_type='test')
    print("Test data: x:"+ str(test_data.x.shape)+ ' y:'+str(test_data.y.shape))

    for fold in range(splits):
        checkpoint_dir = './cross_validation/' + model_name+ '_fold'+ str(fold)
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        if fold == 0:
            predictions = model.predict(test_data.x,verbose=2)
        else:
            predictions = np.concatenate((predictions,model.predict(test_data.x,verbose=2)),axis=1)
    print(predictions.shape)

    avg_predictions = np.mean(predictions, axis=1)
    print(avg_predictions.shape)

    import csv

    output_file = 'result_' + model_name + '.csv'
    with open(output_file, 'wt', newline='', encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(('Id', 'Predicted'))
        for id, predict in enumerate(avg_predictions):
            csv_writer.writerow((id, predict))
    print('Saved to '+'result_' + model_name + '.csv')

def train_model_c_cv():
    seed = 5242
    np.random.seed(seed)
    splits = 8

    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    full_train = Data(data_type='train')
    fold = 0

    for train, test in kfold.split(full_train.x, full_train.y):
        print('fold '+str(fold))
        model_name = 'model_c'
        
        # create model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(90,return_sequences=True),input_shape=(1000,102)))
        model.add(tf.keras.layers.GlobalMaxPooling1D())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=0.1))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        checkpoint_dir = './cross_validation/' + model_name+ '_fold'+ str(fold)
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True)

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            min_delta=0, 
            patience=20, 
            mode='auto', 
            restore_best_weights=True)
    
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(5e-5,0.9,0.999), 
                    metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
        # Train model
        model.fit(full_train.x[train], full_train.y[train], validation_data=(full_train.x[test],full_train.y[test]), 
                epochs=2, batch_size=32, verbose=1,callbacks=[checkpoint_callback,early_stopping_callback],shuffle=True)
        
        fold += 1

def predict_model_c():
    model_name ='model_c'
    splits = 8
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(90,return_sequences=True),input_shape=(1000,102)))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


    test_data = Data(data_type='test')
    print("Test data: x:"+ str(test_data.x.shape)+ ' y:'+str(test_data.y.shape))

    for fold in range(splits):
        checkpoint_dir = './cross_validation/' + model_name+ '_fold'+ str(fold)
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        if fold == 0:
            predictions = model.predict(test_data.x,verbose=2)
        else:
            predictions = np.concatenate((predictions,model.predict(test_data.x,verbose=2)),axis=1)
    print(predictions.shape)

    avg_predictions = np.mean(predictions, axis=1)
    print(avg_predictions.shape)

    import csv

    output_file = 'result_' + model_name + '.csv'
    with open(output_file, 'wt', newline='', encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(('Id', 'Predicted'))
        for id, predict in enumerate(avg_predictions):
            csv_writer.writerow((id, predict))
    print('Saved to '+'result_' + model_name + '.csv')

def combine_results():
    model_a = pd.read_csv(os.getcwd()+'/result_model_a.csv', sep=',', usecols=["Predicted"])
    model_b = pd.read_csv(os.getcwd()+'/result_model_b.csv', sep=',', usecols=["Predicted"])
    model_c = pd.read_csv(os.getcwd()+'/result_model_c.csv', sep=',', usecols=["Predicted"])
    output_file = 'result_final.csv'
    length = model_c.values.shape[0]

    with open(output_file, 'wt', newline='', encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(('Id', 'Predicted'))
        for id in range(length):
            csv_writer.writerow((id, model_a.values[id,0]*0.25+model_b.values[id,0]*0.5+model_c.values[id,0]*0.25))
    print('Saved to result_final.csv')



if __name__ == '__main__':
    train_model_a_boosting()
    predict_model_a()
    train_model_b_cv()
    predict_model_b()
    train_model_c_cv()
    predict_model_c()
    combine_results()

    print("Done")