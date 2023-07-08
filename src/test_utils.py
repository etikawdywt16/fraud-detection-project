import utils
import pandas as pd
import numpy as np
import datetime


def test_label_streamlit():
    mock_dataset = ['Other', 'Major Damage', 'Minor Damage', '?', 'Total Loss', np.nan, 'Trivial Damage']
    mock_dataset = pd.DataFrame(mock_dataset, columns = ['mock_dataset'])

    expected_dataset = ['Major Damage', 'Minor Damage', 'Total Loss', 'Trivial Damage']

    preprocess_data = utils.label_streamlit(mock_dataset, 'mock_dataset')

    assert preprocess_data == expected_dataset


def test_extract_months():
    mock_data_d2 = datetime.datetime(2020, 5, 7)
    mock_data_d1 = datetime.datetime(2023, 7, 8)

    expected_data = 38

    preprocess_data = utils.extract_months(mock_data_d2, mock_data_d1)

    assert preprocess_data == expected_data


def test_extract_years():
    mock_data_d2 = datetime.datetime(2020, 5, 7)
    mock_data_d1 = datetime.datetime(2023, 7, 8)

    expected_data = 3

    preprocess_data = utils.extract_years(mock_data_d2, mock_data_d1)

    assert preprocess_data == expected_data

def test_extract_hours():
    mock_t = datetime.time(hour = 11, minute = 34, second = 56)

    expected_data = 11

    preprocess_data = utils.extract_hours(mock_t)

    assert preprocess_data == expected_data