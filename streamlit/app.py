import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(
    page_title="NITJ Relationship Predictor",
    page_icon="💘",
    layout="wide"
)

st.title("💘 Student Relationship Probability Predictor")
st.markdown("### GDGC AI/ML Inductions Project")


@st.cache_resource
def load_model_package():
  
    current_dir = os.path.dirname(__file__)
    
  
    model_path = os.path.join(current_dir, '..', 'models', 'relationship_model.pkl')
    

    if not os.path.exists(model_path):
        model_path = 'relationship_model.pkl'


    if not os.path.exists(model_path):
        return None
    
    return joblib.load(model_path)


package = load_model_package()


if package is None:
    st.error("🚨 Critical Error: Model file not found!")
    st.info("Please ensure `relationship_model.pkl` is saved in the `models/` folder.")
    st.stop()


model = package['model']
scaler = package['scaler']
model_columns = package['model_columns']


st.sidebar.header("About Model")
st.sidebar.success("✅ Model Loaded Successfully")
st.sidebar.info(
    "This app uses a **Ridge Regression** model trained on NIT Jalandhar student data.\n\n"
    "It predicts the probability (0-100%) of a student being in a relationship."
)


st.subheader("Enter Student Details")


user_input = {}

with st.form("prediction_form"):
    st.write("Adjust the values below to match the student profile:")

    col1, col2, col3 = st.columns(3)
    
    
    for i, col_name in enumerate(model_columns):
        if i % 3 == 0: location = col1
        elif i % 3 == 1: location = col2
        else: location = col3
        
        
        
        if "_" in col_name:
          
            display_name = col_name.split('_')[-1] 
            group_name = col_name.split('_')[0]
            
     
            user_input[col_name] = location.checkbox(f"{group_name}: {display_name}", value=False)
        else:
          
            user_input[col_name] = location.number_input(col_name, value=0.0)

    
    submitted = st.form_submit_button("Predict Probability 🚀", type="primary")


if submitted:
    
    input_df = pd.DataFrame([user_input])
    
  
    input_df = input_df.astype(int)
 
    input_df = input_df.reindex(columns=model_columns, fill_value=0)