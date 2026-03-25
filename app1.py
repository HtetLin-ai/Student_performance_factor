import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

st.set_page_config(page_title="Student Performance AI", layout="wide")
st.title("🎓 Student Exam Score Predictor")
st.markdown("---")

# 2. Create the User Interface (Divided into 3 columns)
col1, col2, col3 = st.columns(3)

with col1:
    st.header("📚 Study Habits")
    Hours_Studied = st.slider("Hours Studied", 0, 50, 20)
    Attendance = st.slider("Attendance %", 0, 100, 90)
    Parental_Involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    Access_to_Resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
    Extracurricular_Activities = st.radio("Extracurricular Activities", ["No", "Yes"])

with col2:
    st.header("🏠 Background")
    Sleep_Hours = st.slider("Sleep Hours", 0, 12, 8)
    Previous_Scores = st.number_input("Previous Test Score", 0, 100, 75)
    Motivation_Level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
    Internet_Access = st.radio("Internet Access", ["No", "Yes"])
    Tutoring_Sessions = st.number_input("Tutoring Sessions", 0, 10, 2)

with col3:
    st.header("🏃 Lifestyle & School")
    Family_Income = st.selectbox("Family Income", ["Low", "Medium", "High"])
    Teacher_Quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
    School_Type = st.radio("School Type", ["Public", "Private"])
    Peer_Influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
    Physical_Activity = st.slider("Physical Activity (hrs/week)", 0, 10, 3)
    Learning_Disabilities = st.radio("Learning Disability", ["No", "Yes"])
    Parental_Education_Level = st.selectbox("Parental Education", ["High School", "College", "Postgraduate"])
    Distance_from_Home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])

# 3. Data Processing Logic
# These must match exactly how you encoded them in the Jupyter Notebook
quality_map = {"Low": 0, "Medium": 1, "High": 2}
edu_map = {"High School": 0, "College": 1, "Postgraduate": 2}
binary_map = {"No": 0, "Yes": 1}
school_map = {"Private": 0, "Public": 1}
dist_map = {"Far": 0, "Moderate": 1, "Near": 2}
peer_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
gender_map = {"Female": 0, "Male": 1}

# 4. Prediction Button
st.markdown("---")
if st.button("🚀 CALCULATE PREDICTED SCORE", use_container_width=True):
    
    # Organize all 19 features in the exact order the model expects
    input_data = [
        Hours_Studied,                          # 0
        Attendance,                     # 1
        quality_map[Parental_Involvement],      # 2
        quality_map[Access_to_Resources],         # 3
        binary_map[Extracurricular_Activities],    # 4
        Sleep_Hours,                          # 5
        Previous_Scores,                    # 6
        quality_map[Motivation_Level],        # 7
        binary_map[Internet_Access],           # 8
        Tutoring_Sessions,                       # 9
        quality_map[Family_Income],            # 10
        quality_map[Teacher_Quality], # Placeholder for Teacher_Quality (can add input if desired) # 11
        school_map[School_Type],        # 12
        peer_map[Peer_Influence],                 # 13
        Physical_Activity,                       # 14
        binary_map[Learning_Disabilities],         # 15
        edu_map[Parental_Education_Level],          # 16
        dist_map[Distance_from_Home],             # 17
    ]
    input_df = pd.DataFrame([input_data], columns=scaler.feature_names_in_)
    # Turn into DataFrame and Scale
    # input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    # Predict
    prediction = model.predict(input_scaled)
    
    # Display Result
    st.balloons()
    st.metric(label="Estimated Exam Score", value=f"{prediction[0]:.2f}/100")
    
    if prediction[0] >= 75:
        st.success("Great! This student is on track for an A/B grade.")
    elif prediction[0] >= 50:
        st.warning("Passing grade, but there is room for improvement.")
    else:
        st.error("Warning: Student is at risk of failing based on current habits.")