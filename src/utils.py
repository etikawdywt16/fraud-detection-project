import pandas as pd
import numpy as np
import yaml
from dateutil import relativedelta
from datetime import date
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# function to load configuration file
def load_config(PATH_CONFIG = 'config/config.yaml'):
    try:
        config = yaml.safe_load(open(PATH_CONFIG))
    except:
        config = yaml.safe_load(open('../config/config.yaml'))
    return config


# function to load train and test dataset
def load_dataset(file_name, config = load_config()):
    try:
        PATH = config['train_test_data']['directory'] + file_name
        file_load = joblib.load(PATH)
    except:
        PATH = '../' + config['train_test_data']['directory'] + file_name
        file_load = joblib.load(PATH)
    return file_load


# function to dump dataset
def dump_dataset(to_dump, file_name, config = load_config()):
    try:
        joblib.dump(to_dump, config['train_test_data']['directory'] + file_name)
    except:
        joblib.dump(to_dump, '../' + config['train_test_data']['directory'] + file_name)


# function to dump model
def dump_model(to_dump, model_name, config = load_config()):
    try:
        joblib.dump(to_dump, config['final_model']['model_directory'] + model_name)
    except:
        joblib.dump(to_dump, '../' + config['final_model']['model_directory'] + model_name)


# function to return categories in columns
def label_streamlit(dataset, columns):
    label = dataset[columns].unique().tolist()
    label = [value for value in label if str(value) != '?']
    label = [value for value in label if str(value) != 'nan']
    label = [value for value in label if str(value) != 'Other']

    return np.sort(label).tolist()


# function to extract difference months between two date
def extract_months(d2,
                   d1 = date.today()):
    delta = relativedelta.relativedelta(d1, d2)

    return delta.years * 12 + delta.months


# function to extract difference years between two date
def extract_years(d2,
                  d1 = date.today()):
    delta = relativedelta.relativedelta(d1, d2)

    return delta.years


# function to extract hours from time
def extract_hours(t):

    return t.hour


# DATA DEFENSE
# function to check data requirements
def check_data(input_data, config = load_config()):
    # check range data and missing value
    # policy number
    assert len(input_data['policy_number'][0]) == config['data_defense']['policy_number']['length']\
        or input_data['policy_number'][0] != '',\
        'policy number must be 6 digit length and cannot be empty.'

    # months as customer
    assert input_data['months_as_customer'][0] >= config['data_defense']['months_as_customer']['min_value'],\
        'check inception date.'
    
    # policy state
    assert input_data['policy_state'][0] in config['data_defense']['policy_state']['value'] or\
        input_data['policy_state'][0] != '',\
        f"policy state must be in list {config['data_defense']['policy_state']['value']}, and cannot be empty."
    
    # policy csl
    assert input_data['policy_csl'][0] in config['data_defense']['policy_csl']['value'] or\
        input_data['policy_csl'][0] != '',\
        f"policy csl must be in list {config['data_defense']['policy_csl']['value']}, and cannot be empty."

    # policy deductable
    assert input_data['policy_deductable'][0] > config['data_defense']['policy_deductable']['min_value'],\
        f"policy deductable must be greater than {config['data_defense']['policy_deductable']['min_value']}"
    
    # umbrella limit
    assert input_data['umbrella_limit'][0] >= config['data_defense']['umbrella_limit']['min_value'],\
        f"umbrella limit must be greater than or equal to {config['data_defense']['umbrella_limit']['min_value']}"
    
    # annual premium
    assert input_data['policy_annual_premium'][0] > config['data_defense']['policy_annual_premium']['min_value'],\
        f"annual premium must be greater than {config['data_defense']['policy_annual_premium']['min_value']}"

    # capital gains
    assert input_data['capital-gains'][0] >= config['data_defense']['capital-gains']['min_value'],\
        f"capital gains must be greater than or equal to {config['data_defense']['capital-gains']['min_value']}"
    
    # capital loss
    assert input_data['capital-loss'][0] <= config['data_defense']['capital-loss']['max_value'],\
        f"capital loss must be lower than or equal to {config['data_defense']['capital-loss']['max_value']}"
    
    # insured age
    assert input_data['age'][0] >= config['data_defense']['insured_age']['min_value'],\
        'check insured birth of date'
    
    # insured sex
    assert input_data['insured_sex'][0] in config['data_defense']['insured_sex']['value'] or\
        input_data['insured_sex'][0] != '',\
        f"insured sex must be in list {config['data_defense']['insured_sex']['value']}, and cannot be empty"

    # insured zip
    assert input_data['insured_zip'][0] == config['data_defense']['insured_zip']['length'] or\
        len(input_data['insured_zip'][0]) != '',\
        'insured zip must be 6 digit length and cannot be empty.'
    
    # insured occupation
    assert input_data['insured_occupation'][0] == config['data_defense']['insured_occupation']['value'] or\
        input_data['insured_occupation'][0] != '',\
        f"insured occupation must be in list {config['data_defense']['insured_occupation']['value']} or Other, and cannot be empty."  
    
    # insured hobbies
    assert input_data['insured_hobbies'][0] != '',\
        f"insured hobbies cannot be empty."
    
    # insured education level
    assert input_data['insured_education_level'][0] == config['data_defense']['insured_education_level']['value'] or\
        input_data['insured_education_level'][0] != '',\
        f"insured education level must be in list {config['data_defense']['insured_education_level']['value']}, and cannot be empty."
    
    # insured relationship
    assert input_data['insured_relationship'][0] == config['data_defense']['insured_relationship']['value'] or\
        input_data['insured_relationship'][0] != '',\
        f"insured relationship level must be in list {config['data_defense']['insured_relationship']['value']}, and cannot be empty."
    
    # incident state
    assert input_data['incident_state'][0] != '',\
        'incident state cannot be empty.'
    
    # incident city
    assert input_data['incident_city'][0] != '',\
        'incident city cannot be empty.'

    # incident location
    assert input_data['incident_location'][0] != '',\
        'incident location cannot be empty.'
    
    # incident type
    assert input_data['incident_type'][0] in config['data_defense']['incident_type']['value'] or\
        input_data['incident_type'][0] != '',\
        f"incident type must be in list {config['data_defense']['incident_type']['value']} or Other, and cannot be empty."

    # collision type
    assert input_data['collision_type'][0] in config['data_defense']['collision_type']['value'] or\
        input_data['collision_type'][0] != '',\
        f"collision type must be in list {config['data_defense']['collision_type']['value']}, or Other, and cannot be empty."

    # incident severity
    assert input_data['incident_severity'][0] in config['data_defense']['incident_severity']['value'] or\
        input_data['incident_severity'][0] != '',\
        f"incident severity must be in list {config['data_defense']['incident_severity']['value']} or Other, and cannot be empty."

    # number of vehicle involved
    assert input_data['number_of_vehicles_involved'][0] >= config['data_defense']['number_of_vehicles_involved']['min_value'],\
        'number of vehicle involved must be non-negative'
    
    # number of bodily injuries
    assert input_data['bodily_injuries'][0] >= config['data_defense']['bodily_injuries']['min_value'],\
        'number of bodily injuries must be non-negative'
    
    # number of witnesses
    assert input_data['witnesses'][0] >= config['data_defense']['witnesses']['min_value'],\
        'number of witnesses must be non-negative'
    
    # property damage
    assert input_data['property_damage'][0] in config['data_defense']['property_damage']['value'] or\
        input_data['property_damage'][0] != '',\
        f"property damage must be in list {config['data_defense']['property_damage']['value']}, and cannot be empty."
    
    # automotive makers
    assert input_data['auto_make'][0] != '',\
        'automotive makers cannot be empty'

    # automotive model
    assert input_data['auto_model'][0] != '',\
        'automotive model cannot be empty'
    
    # automotive year
    assert input_data['auto_year'][0] != '',\
        'automotive year cannot be empty'

    # authorities contacted
    assert input_data['authorities_contacted'][0] in config['data_defense']['authorities_contacted']['value'] or\
         input_data['authorities_contacted'][0] != '',\
        f"authorities contacted must be in list {config['data_defense']['authorities_contacted']['value']}, or Other, and cannot be empty."

    # police report available
    assert input_data['police_report_available'][0] in config['data_defense']['police_report_available']['value'] or\
         input_data['police_report_available'][0] != '',\
        f"police report available must be in list {config['data_defense']['police_report_available']['value']}, and cannot be empty."
    
    # injury claim
    assert input_data['injury_claim'][0] >= config['data_defense']['injury_claim']['min_value'],\
        'injury claim must be positive'
    
    # property claim
    assert input_data['property_claim'][0] >= config['data_defense']['property_claim']['min_value'],\
        'property claim must be positive'
    
    # vehicle claim
    assert input_data['vehicle_claim'][0] >= config['data_defense']['vehicle_claim']['min_value'],\
        'vehicle claim must be positive'
    

    # check data types
    #assert input_data.select_dtypes('object').columns.to_list() == config['data_source']['cat_features'],\
    #    f"{set(config['data_source']['cat_features'])} {set(input_data.select_dtypes('object').columns.to_list())}\
    #        must be string."
    
    assert input_data.select_dtypes('int').columns.to_list() == config['data_source']['num_int_features'],\
        'an error occurs in int column(s).'
    
    assert input_data.select_dtypes('float').columns.to_list() == config['data_source']['num_float_features'],\
        'an error occurs in float column(s).'


# PREPROCESSING
# function to remove irrelevant features
def drop_features(X, config = load_config()):
    X = X.drop(config['preprocess']['drop_features'], axis = 1)

    return X


# function to fixing data type
def fix_data_type(X, config = load_config()):
    X[config['preprocess']['cat_features']] = X[config['preprocess']['cat_features']].astype(object)
    X[config['preprocess']['num_int_features']] = X[config['preprocess']['num_int_features']].astype(int)
    X[config['preprocess']['num_float_features']] = X[config['preprocess']['num_float_features']].astype(float)

    return X


# function to handling missing values
def handle_missing_value(X, config = load_config()):
    X[config['preprocess']['cat_features']] = X[config['preprocess']['cat_features']].replace('?', None)
    X[config['preprocess']['cat_features']] = X[config['preprocess']['cat_features']].fillna('UNKNOWN')

    return X


# function to return clean input dataset
def preprocess_input(X, config = load_config()):
    X = drop_features(X, config = config)
    X = fix_data_type(X, config = config)
    X = handle_missing_value(X, config = config)

    return X


# function to encoding categorical data
def ohe_input(X, config = load_config()):
    X_train = load_dataset('X_train.pkl')
    X_train_preprocess = preprocess_input(X_train)

    ohe = OneHotEncoder(handle_unknown = 'ignore')
    ohe.fit(X_train_preprocess[config['preprocess']['cat_features']])
    
    ohe_data_raw = ohe.transform(X[config['preprocess']['cat_features']]).toarray()
    X_index = X.index
    X_features = ohe.get_feature_names_out()

    X_ohe = pd.DataFrame(ohe_data_raw, index = X_index, columns = X_features)

    return X_ohe


# funtion to concat all features
def clean_input(X_preprocess, X_ohe, config = load_config()):
    X_clean = pd.concat([X_preprocess[config['preprocess']['num_features']], X_ohe], axis=1)

    return X_clean


# function to encoding target data
def ohe_output(y):
    y_ohe = y.replace({'Y': 1, 'N': 0})
    
    return y_ohe


# function to resampling data
def resampling(X, y, sampler, config = load_config()):
    if sampler == None:
        X_sam, y_sam = X, y

    else:
        sampler = sampler(random_state = config['feature_eng']['sampler_random_state'])
        X_sam, y_sam = sampler.fit_resample(X, y)

    return X_sam, y_sam


# MODELLING
# function to training model
def train_model(X_train, y_train, config = load_config()):
    param = config['final_model']['parameter']
    dt = DecisionTreeClassifier(**param)
    dt.fit(X_train, y_train)
    return dt

# function to show metrics evaluation
def evaluation_model(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    report = classification_report(y_true = y_test,
                                   y_pred = y_test_pred)
    print(report)