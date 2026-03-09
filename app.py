import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('titanic_model.pkl')

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict if they would have survived.")

# Layout: Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3], help="1 = First, 2 = Second, 3 = Third")
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 30)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)

with col2:
    parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
    fare = st.slider("Fare Paid", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

embark_map = {"C":0,"Q":1,"S":2}

# Preprocessing logic (matching the training format)
if st.button("Predict Survival"):
    # Create a dataframe for the input
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [1 if sex == 'male' else 0],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked' : embark_map[embarked]
    })

    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    # Display results
    if prediction[0] == 1:
        st.success(f"The passenger likely SURVIVED. (Probability: {probability:.2%})")
    else:
        st.error(f"The passenger likely PERISHED. (Probability: {1-probability:.2%})")
