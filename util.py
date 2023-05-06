import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_vehicle_data(vehicles_path='data'):
  csv_path = os.path.join(vehicles_path, "project_data.csv")
  return pd.read_csv(csv_path)


def remove_vehicle_data_noise(vehicles):
  # Removing outlier vehicle prices that are dramatically outside standard deviation
  vehicles.drop(vehicles[vehicles['selling_price'] >= 1650000].index, inplace = True)

  # Removing vehicles older than 2000 due to sparse inconsistent data
  vehicles.drop(vehicles[vehicles['year'] < int("2000")].index, inplace = True)

  # Removing sparse high mileage vehicles
  vehicles.drop(vehicles[vehicles['km_driven'] >= 240000].index, inplace = True)

  # Removing 'Test Drive Car' instances since it's NOT numerically related to other owner types
  # AND NOT BEING USED FOR ONE HOT ENCODING
  vehicles.drop(vehicles[vehicles['owner'] == 'Test Drive Car'].index, inplace = True)

  # Remove 'Electric' 'fuel' type since only 1 instance
  vehicles.drop(vehicles[vehicles['fuel'] == 'Electric'].index, inplace = True)
  

def convert_categories_to_integers(vehicles):
  # Creating numerical value for number of owner strings
  current_owner_strings = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner']
  new_owner_strings = ['1', '2', '3', '4']
  vehicles['owner'] = vehicles['owner'].replace(current_owner_strings, new_owner_strings)
  vehicles['owner'] = vehicles['owner'].astype('int64') # Convert type to integers:

  # 0-indexing to convert category column's to be one hot encoded 
  current_owner_strings = ['Petrol', 'Diesel', 'CNG', 'LPG']
  new_strings = ['0', '1', '2', '3']
  vehicles['fuel'] = vehicles['fuel'].replace(current_owner_strings, new_strings)
  vehicles['fuel'] = vehicles['fuel'].astype('int64')

  current_owner_strings = ['Individual', 'Dealer', 'Trustmark Dealer']
  new_strings = ['0', '1', '2']
  vehicles['seller_type'] = vehicles['seller_type'].replace(current_owner_strings, new_strings)
  vehicles['seller_type'] = vehicles['seller_type'].astype('int64')

  current_owner_strings = ['Manual', 'Automatic']
  new_strings = ['0', '1']
  vehicles['transmission'] = vehicles['transmission'].replace(current_owner_strings, new_strings)
  vehicles['transmission'] = vehicles['transmission'].astype('int64')

  return vehicles

def normalize_numerical_data_attributes(data):
  numerical_data = data[['year', 'km_driven', 'owner']].copy()
  normaled = (numerical_data - numerical_data.mean()) / numerical_data.std()
  data.drop(['year', 'km_driven', 'owner'], axis=1, inplace = True)

  data['year'] = normaled['year']
  data['km_driven'] = normaled['km_driven']
  data['owner'] = normaled['owner']

  return data
  

# Helper function to create a new data set containing 'data_set'
# elements in 'column_name_str' of data set and remove
# that column from 'data_set' input
def extract_data_column_inplace(data_set, column_name_str):
  data_column = data_set[[column_name_str]].copy()
  data_set.drop([column_name_str], axis=1, inplace = True)

  return data_column


def create_neural_net_partitions_train(train_set):
  # Copying and deleting data for one hot coding portion of model
  train_set_one_hot_fuel = extract_data_column_inplace(train_set, 'fuel')
  train_set_one_hot_seller_type = extract_data_column_inplace(train_set, 'seller_type')
  train_set_one_hot_transmission = extract_data_column_inplace(train_set, 'transmission')

  # Extracting name attribute column
  train_set_names = extract_data_column_inplace(train_set, 'name')

  # Extracting selling price column
  train_set_labels = extract_data_column_inplace(train_set, 'selling_price')

  ''' Converting partittions to tensors '''
  train_set_tensor = tf.convert_to_tensor(train_set)
  train_set_one_hot_fuel_tensor = tf.convert_to_tensor(train_set_one_hot_fuel)
  train_set_one_hot_seller_type_tensor = tf.convert_to_tensor(train_set_one_hot_seller_type)
  train_set_one_hot_transmission_tensor = tf.convert_to_tensor(train_set_one_hot_transmission)
  train_set_names_tensor = tf.convert_to_tensor(train_set_names)
  train_set_labels_tensor = tf.convert_to_tensor(train_set_labels)

  return train_set_names, train_set_tensor, train_set_one_hot_fuel_tensor, train_set_one_hot_seller_type_tensor, train_set_one_hot_transmission_tensor, train_set_names_tensor, train_set_labels_tensor


def create_neural_net_partitions_test(test_set):
  test_set_one_hot_fuel = extract_data_column_inplace(test_set, 'fuel')
  test_set_one_hot_seller_type = extract_data_column_inplace(test_set, 'seller_type')
  test_set_one_hot_transmission = extract_data_column_inplace(test_set, 'transmission')

  test_set_names = extract_data_column_inplace(test_set, 'name')

  test_set_labels = extract_data_column_inplace(test_set, 'selling_price')

  test_set_tensor = tf.convert_to_tensor(test_set)
  test_set_one_hot_fuel_tensor = tf.convert_to_tensor(test_set_one_hot_fuel)
  test_set_one_hot_seller_type_tensor = tf.convert_to_tensor(test_set_one_hot_seller_type)
  test_set_one_hot_transmission_tensor = tf.convert_to_tensor(test_set_one_hot_transmission)
  test_set_names_tensor = tf.convert_to_tensor(test_set_names)
  test_set_labels_tensor = tf.convert_to_tensor(test_set_labels)

  return test_set_names, test_set_tensor, test_set_one_hot_fuel_tensor, test_set_one_hot_seller_type_tensor, test_set_one_hot_transmission_tensor, test_set_names_tensor, test_set_labels_tensor


def load_preprocessed_data():
  vehicles = load_vehicle_data()

  remove_vehicle_data_noise(vehicles)

  converted_vehicles = convert_categories_to_integers(vehicles)

  converted_vehicles.reset_index(inplace=True)

  # Drop column old indexes created by calling 'reset_index'
  # converted_vehicles.drop(converted_vehicles['index'].index, inplace = True)
  converted_vehicles.drop(['index'], axis=1, inplace = True)

  normalize_numerical_data_attributes(converted_vehicles)

  # Split data
  train_set, test_set = train_test_split(vehicles, test_size=0.2, random_state=42)

  return train_set, test_set
