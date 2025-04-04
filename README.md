# Campus-placement-prediction-using-ml by rauhsnabyahut7786@gmail.com
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Sample Data (including new features)
data = {
    "Tenth_Percentage": [85, 90, 80, 75, 95, 88, 82, 91, 87, 78],
    "Tenth_Board": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "Twelfth_Percentage": [80, 85, 78, 82, 90, 87, 80, 86, 83, 79],
    "Twelfth_Board": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "Stream": [0, 1, 0, 2, 0, 1, 0, 3, 0, 1],
    "Graduation_CGPA": [7.5, 8.0, 7.0, 6.5, 8.5, 7.8, 7.2, 8.2, 7.9, 7.3],
    "Backlogs": [0, 1, 0, 2, 0, 0, 1, 2, 0, 1],
    "Internships": [1, 2, 1, 0, 2, 1, 0, 1, 1, 2],
    "Java": [2, 3, 1, 1, 3, 2, 1, 3, 2, 1],
    "Python": [3, 3, 2, 1, 3, 3, 2, 3, 3, 2],
    "Hackathon": [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],  # New feature
    "Certificates": [2, 3, 1, 1, 3, 2, 1, 3, 2, 2],  # New feature
    "Communication_Skills": [2, 3, 1, 1, 3, 2, 1, 2, 3, 2],  # New feature
    "Placed": [1, 1, 0, 0, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Train-Test Split
X = df.drop(columns=["Placed"])
y = df["Placed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Set Page Config
st.set_page_config(page_title="Campus Placement Prediction", page_icon="🎓", layout="wide")

# Function to Set Background Image
def set_background():
    image_path = "/mnt/data/inbox_3928966_90d3853c1696de38e8fde2b7b994fac2_index.jpg"
    
    bg_style = f"""
    <style>
    .stApp {{
        background-image: url("{image_path}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# Call Function to Set Background
set_background()

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔑 Student Login", "📊 Result"])

# Home Page
if page == "🏠 Home":
    st.markdown("<h1 style='text-align: center; font-weight: bold;'>🎓 Campus Placement Prediction</h1>", unsafe_allow_html=True)
    st.write("Campus placement is a recruitment process where companies visit colleges to hire students for jobs or internships. It provides students with career opportunities, industry exposure, and a smooth transition from academics to the professional world. A well-structured placement program enhances employability and helps organizations find skilled talent efficiently.")
    st.write("Start by entering your details under **Student Login**.")

# Student Login Page
elif page == "🔑 Student Login":
    st.subheader("Enter Your Details")
    tenth_percentage = st.number_input("🎓 10th Percentage", 0.0, 100.0, 75.0)
    tenth_board = st.selectbox("🏫 10th Board", ["State", "Central"])
    twelfth_percentage = st.number_input("📚 12th Percentage", 0.0, 100.0, 80.0)
    twelfth_board = st.selectbox("🏛 12th Board", ["State", "Central"])
    stream = st.selectbox("🎓 Stream", ["CSE", "ME", "CE", "EE"])
    graduation_cgpa = st.number_input("🎓 Graduation CGPA", 0.0, 10.0, 7.0)
    backlogs = st.number_input("📑 Backlogs", 0, 10, 0)
    internships = st.number_input("💼 Internships", 0, 5, 1)

    # New features for skill-based inputs
    hackathon = st.number_input("🎯 Hackathon (1 if attended)", 0, 1, 0)
    certificates = st.number_input("📜 Certificates (minimum 2 required)", 0, 10, 2)
    communication_skills = st.selectbox("🗣 Communication Skills", ["Beginner", "Intermediate", "Advanced"])
    
    coding_skills = st.multiselect("💻 Select Coding Skills", ["Java", "Python", "C++", "SQL", "Data Structures"])
    skill_levels = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}

    # Initialize skill scores to 0 for all coding skills
    skill_scores = {"Java": 0, "Python": 0, "C++": 0, "SQL": 0, "Data Structures": 0}
    
    for skill in coding_skills:
        skill_scores[skill] = skill_levels[st.selectbox(f"{skill} Skill Level", list(skill_levels.keys()))]

    if st.button("🔮 Predict Placement"):
        user_data = {
            "Tenth_Percentage": tenth_percentage,
            "Tenth_Board": 1 if tenth_board == "Central" else 0,
            "Twelfth_Percentage": twelfth_percentage,
            "Twelfth_Board": 1 if twelfth_board == "Central" else 0,
            "Stream": ["CSE", "ME", "CE", "EE"].index(stream),
            "Graduation_CGPA": graduation_cgpa,
            "Backlogs": backlogs,
            "Internships": internships,
            "Hackathon": hackathon,
            "Certificates": certificates,
            "Communication_Skills": skill_levels[communication_skills],
        }
        user_data.update(skill_scores)  # Ensure all skills are present

        # Convert user input into DataFrame
        user_df = pd.DataFrame([user_data])

        # Match feature order to model training data
        user_df = user_df[X.columns]

        # Save user data in session
        st.session_state.user_data = user_df
        st.success("✅ Data saved! Go to '📊 Result' to view the prediction.")

# Result Page
elif page == "📊 Result":
    st.markdown("<h2 style='color: red; font-family: \"Times New Roman\"; font-size: 18px;'>Prediction Results</h2>", unsafe_allow_html=True)
    if "user_data" not in st.session_state:
        st.warning("⚠️ No input data found! Please enter your details first.")
    else:
        user_df = st.session_state.user_data

        # Get Communication Skills
        communication_skills_level = user_df["Communication_Skills"][0]

        # Check if all critical inputs are above 70%
        if (user_df["Tenth_Percentage"][0] >= 70 and 
            user_df["Twelfth_Percentage"][0] >= 70 and 
            user_df["Graduation_CGPA"][0] >= 7):
            st.success("🎉 Congratulations! You are highly likely to be placed.")
        else:
            # Perform prediction only if communication skill is Intermediate or above
            if communication_skills_level >= 2:
                # Make prediction
                prediction = model.predict(user_df)[0]
                probability = model.predict_proba(user_df)[0][1]

                if prediction == 1:
                    st.success(f"🎉 Likely to be placed! (Probability: {probability * 100:.2f}%)")
                else:
                    st.error(f"❌ Placement is uncertain. (Probability: {probability * 100:.2f}%)")
            else:
                st.error("❌ Placement prediction not possible due to insufficient communication skills.")

        # Feature Importance Chart
        st.subheader("Feature Importance")
        fig, ax = plt.subplots()
        sns.barplot(x=model.feature_importances_, y=X.columns, ax=ax)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

