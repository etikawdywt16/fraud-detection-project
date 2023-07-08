import streamlit as st
from PIL import Image
import requests
import pandas as pd
from datetime import date
import json
from utils import *


config = load_config()

file_name = config['data_source']['directory'] + config['data_source']['file_name']
dataset = pd.read_csv(file_name)

image = Image.open('assets/fraud_claim_image.jpg')
st.image(image, use_column_width  = 'always')

st.title('CLAIM FRAUD DETECTION')
st.subheader('Please input claim detail')

# create form
with st.form(key = 'POLICY DETAIL'):
    
    # policy detail
    policy_number = st.text_input(
        label = 'Policy Number'
    )

    policy_bind_date = st.date_input(
        label = 'Inception Date',
        max_value=date.today()
    )

    policy_state = st.selectbox(
        label = 'Policy State',
        options = [''] + label_streamlit(dataset = dataset, columns = 'policy_state')
    )

    policy_csl = st.selectbox(
        label = 'Policy CSL',
        options = [''] + label_streamlit(dataset = dataset, columns = 'policy_csl')
    )

    policy_deductable = st.number_input(
        label = 'Policy Deductable',
        min_value=0.0
    )

    umbrella_limit = st.number_input(
        label = 'Umbrella Limit',
        min_value=0.0
    )

    policy_annual_premium = st.number_input(
        label = 'Policy Annual Premium',
        min_value=0.0
    )

    capital_gains = st.number_input(
        label = 'Capital Gains',
        min_value=0.0
    )

    capital_loss = st.number_input(
        label = 'Capital Loss',
        max_value=0.0
    )


    # insured detail
    insured_bod = st.date_input(
        label = 'Insured Birth of Date',
        max_value=date.today()
    )

    insured_sex = st.selectbox(
        label = 'Insured Sex',
        options = [''] + label_streamlit(dataset = dataset, columns = 'insured_sex')
    )

    insured_zip = st.text_input(
        label = 'Insured ZIP'
    )

    insured_occupation = st.selectbox(
        label = 'Insured Occupation',
        options = [''] + label_streamlit(dataset = dataset, columns = 'insured_occupation') + ['Other']
    )

    insured_hobbies = st.selectbox(
        label = 'Insured Hobbies',
        options = [''] + label_streamlit(dataset = dataset, columns = 'insured_hobbies') + ['Other']
    )
    
    insured_education_level = st.selectbox(
        label = 'Insured Education Level',
        options = [''] + label_streamlit(dataset = dataset, columns = 'insured_education_level') + ['Other']
    )

    insured_relationship = st.selectbox(
        label = 'Insured Relationship',
        options = [''] + label_streamlit(dataset = dataset, columns = 'insured_relationship')
    )


    # claim detail
    incident_date = st.date_input(
        label = 'Incident Date',
        max_value=date.today()
    )

    incident_time = st.time_input(
        label = 'Incident Time',
        step = 60
    )

    incident_state = st.selectbox(
        label = 'Incident State',
        options = [''] + label_streamlit(dataset = dataset, columns = 'incident_state') + ['Other']
    )

    incident_city = st.selectbox(
        label = 'Incident City',
        options = [''] + label_streamlit(dataset = dataset, columns = 'incident_city') + ['Other']
    )

    incident_location = st.text_input(
        label = 'Incident Location'
    )

    incident_type = st.selectbox(
        label = 'Incident Type',
        options = [''] + label_streamlit(dataset = dataset, columns = 'incident_type') + ['Other']
    )

    collision_type = st.selectbox(
        label = 'Collision Type',
        options = [''] + label_streamlit(dataset = dataset, columns = 'collision_type') + ['Other']
    )

    incident_severity = st.selectbox(
        label = 'Incident Severity',
        options = [''] + label_streamlit(dataset = dataset, columns = 'incident_severity') + ['Other']
    )

    number_of_vehicles_involved = st.number_input(
        label = 'Number of Vehicles Involved',
        step = 1,
        min_value=0
    )

    bodily_injuries = st.number_input(
        label = 'Number of Bodily Injuries',
        step = 1,
        min_value=0
    )

    witnesses = st.number_input(
        label = 'Number of Witnesses',
        step = 1,
        min_value=0
    )

    property_damage = st.selectbox(
        label = 'Property Damage',
        options = [''] + label_streamlit(dataset = dataset, columns = 'property_damage')
    )

    auto_make = st.selectbox(
        label = 'Automotive Makers',
        options = [''] + label_streamlit(dataset = dataset, columns = 'auto_make') + ['Other']
    )

    auto_model = st.selectbox(
        label = 'Automotive Model',
        options = [''] + label_streamlit(dataset = dataset, columns = 'auto_model') + ['Other']
    )

    auto_year = st.selectbox(
        label = 'Automotive Year',
        options = [''] + list(range(min(dataset['auto_year']), 2024, 1))
    )

    authorities_contacted = st.selectbox(
        label = 'Authorities Contacted',
        options = [''] + label_streamlit(dataset = dataset, columns = 'authorities_contacted') + ['Other']
    )

    police_report_available = st.selectbox(
        label = 'Police Report Available',
        options = [''] + label_streamlit(dataset = dataset, columns = 'police_report_available')
    )

    injury_claim = st.number_input(
        label = 'Injury Claim',
        min_value=0.0
    )
    
    property_claim = st.number_input(
        label = 'Property Claim',
        min_value=0.0
    )

    vehicle_claim = st.number_input(
        label = 'Vehicle Claim',
        min_value=0.0
    )
    
    submit = st.form_submit_button(label="Predict")

    if submit:
    
    # collect data
        form_data = {
            'months_as_customer': extract_months(d2 = policy_bind_date),
            'age': extract_years(d2 = insured_bod),
            'policy_number': policy_number,
            'policy_bind_date': json.dumps(policy_bind_date, default = str),
            'policy_state': policy_state,
            'policy_csl': policy_csl,
            'policy_deductable': policy_deductable,
            'policy_annual_premium': policy_annual_premium,
            'umbrella_limit': umbrella_limit,
            'insured_zip': insured_zip,
            'insured_sex': insured_sex,
            'insured_education_level': insured_education_level,
            'insured_occupation': insured_occupation,
            'insured_hobbies': insured_hobbies,
            'insured_relationship': insured_relationship,
            'capital-gains': capital_gains,
            'capital-loss': capital_loss,
            'incident_date': json.dumps(incident_date, default = str),
            'incident_type': incident_type,
            'collision_type': collision_type,
            'incident_severity': incident_severity,
            'authorities_contacted': authorities_contacted,
            'incident_state': incident_state,
            'incident_city': incident_city,
            'incident_location': incident_location,
            'incident_hour_of_the_day': extract_hours(incident_time),
            'number_of_vehicles_involved': number_of_vehicles_involved,
            'property_damage': property_damage,
            'bodily_injuries': bodily_injuries,
            'witnesses': witnesses,
            'police_report_available': police_report_available,
            'total_claim_amount': injury_claim + property_claim + vehicle_claim,
            'injury_claim': injury_claim,
            'property_claim': property_claim,
            'vehicle_claim': vehicle_claim,
            'auto_make': auto_make,
            'auto_model': auto_model,
            'auto_year': str(auto_year)
        }

        # sending data to api
        with st.spinner('Sending data to prediction server ...'):
            res = requests.post('http://api:8000/predict', json = form_data).json()

        if res['status'] == 200:
            st.success(f'FRAUD DETECTION: {res["prediction"]}')
        else:
            st.error(f'ERROR DETECTION: {res}')