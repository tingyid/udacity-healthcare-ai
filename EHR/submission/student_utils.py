import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        reduced_df: pandas dataframe, output dataframe with joined generic drug name
    '''
    
    reduced_df = pd.merge(df, ndc_df, left_on='ndc_code', right_on='NDC_Code', how='left')
    reduced_df.rename(columns={'Proprietary Name':'generic_drug_name'},inplace=True)

    column_lists = df.columns.tolist()
    column_lists.append('generic_drug_name')

    reduced_df = reduced_df[column_lists]
    print('done')
    return reduced_df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    
    df.sort_values(['patient_nbr','encounter_id'])
    first_encounters = df.groupby('patient_nbr')['encounter_id'].head(1).values
    first_encounter_df = df[df['encounter_id'].isin(first_encounters)]

    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split into roughly 60/20/20 for train,
        validation, and test. 
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''

    """ 
    df = df.iloc[np.random.permutation(len(df))]
    unique_patients = df[patient_key].unique()
    total_patients = len(unique_patients)
    
    # Split data into training/test (80/20)
    sample_size = round(total_patients * 0.8)
    train_val = df[df[patient_key].isin(unique_patients[:sample_size])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_patients[sample_size:])].reset_index(drop=True)
    
    # Split training data into training/validation (60/20)
    train_size = round(sample_size * 0.75) 
    train = train_val[train_val[patient_key].isin(unique_patients[:train_size])].reset_index(drop=True)
    validation = train_val[train_val[patient_key].isin(unique_patients[train_size:])].reset_index(drop=True)
    return train, validation, test

    """
    train, test = train_test_split(df, test_size=0.2,)
    train, val = train_test_split(train, test_size=0.25)

    print('{} train examples: {:0.2f}%'.format(len(train), len(train)/len(df)*100))
    print('{} validation examples: {:0.2f}%'.format(len(val), len(val)/len(df)*100))
    print('{} test examples: {:0.2f}%'.format(len(test), len(test)/len(df)*100))

    return train, val, test
    

#Question 7
def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(key=c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_feature_column)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list


#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std

def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field
    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)
    return tf.feature_column.numeric_column(key=col, default_value = default_value, normalizer_fn=normalizer, dtype=tf.float64)


#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    
    student_binary_prediction = df[col].apply(lambda x: 1 if x>=5 else 0)
    return student_binary_prediction

## Other util functions
def check_null_values(df): 
	null_df = pd.DataFrame({'columns': df.columns, 
                            'percent_null': df.isnull().sum() * 100 / len(df), 
                            'percent_zero': df.isin([0]).sum() * 100 / len(df),
                            'percent_question': df.isin(['?']).sum() * 100 / len(df)
                        })
	return null_df
