import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

model = joblib.load('lr.pkl')
st.title("HDB Resale Price Prediction")
towns = ['Bedok', 'Punggol', 'Tampines']
flat_types = ['2 room', '3 room', '4 room']
storey_range = ['01 TO 03', '04 TO 06', '07 TO 09']



if st.button("predict HDP Price"):
    town_selected = st.selectbox("Select Town", towns)
    flat_type_selected = st.selectbox("Select Flat Type", flat_types)
    storey_range_selected = st.selectbox("Select Storey", storey_range)
    floor_area_selected = st.selectbox("Select Floor Area", min_value = 20, max_value = 700, value = 70)

    input_data = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range],
        'floor_area': [floor_area_selected],
    })
    
    df_input = pd.DataFrame({
        'town': [town_selected],
        'flat_type': [flat_type_selected],
        'storey_range': [storey_range],
        'floor_area': [floor_area_selected],
    })

    df_input = pd.get_dummies(df_input,
                            columns=['town', 'flat_type', 'storey_range'])

    df_input = df_input.reindex(columns=model.feature_names_in_,
                                fill_value=0)

    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Resale Price: ${y_unseen_pred:,.2f}")

st.markdown(
    f"""
    <style>
        .stApp ({
        background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcfVjhM1jQlOs139h4BIyoM8c8WFaApDIlPQ&s");
        background-size: cover;
        })
    </style>
    """,
    unsafe_allow_html=True
)

