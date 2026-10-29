
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Load saved model and symptom names
with open("disease_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("symptom_columns.pkl", "rb") as file:
    symptom_columns = pickle.load(file)

# Header
st.title("🩺 AI-Based Disease Prediction System")
st.write(
    "An educational machine-learning prototype that explores "
    "relationships between reported symptoms and dataset disease labels."
)

st.warning(
    "Educational use only — this tool is not clinically validated "
    "and must not be used for diagnosis or treatment decisions."
)

# Sidebar information
with st.sidebar:
    st.header("About the Project")
    st.write("**Algorithm:** Random Forest")
    st.write("**Symptom features:**", len(symptom_columns))
    st.write("**Disease classes:**", len(model.classes_))
    st.divider()
    st.caption("Developed as an academic AI/ML project.")

# Main layout
left, right = st.columns([2, 1])

with left:
    st.subheader("1. Enter Symptoms")

    selected_symptoms = st.multiselect(
        "Search for symptoms and select all that apply:",
        options=symptom_columns,
        format_func=lambda s: s.replace("_", " ").strip().title()
    )

    st.caption(f"Selected symptoms: {len(selected_symptoms)}")

    predict_button = st.button(
        "🔍 Predict Disease Label",
        type="primary",
        use_container_width=True
    )

with right:
    st.subheader("2. How It Works")
    st.write("1. Select symptoms.")
    st.write("2. The app converts them into model inputs.")
    st.write("3. The trained model returns a predicted label.")

if predict_button:
    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        input_data = pd.DataFrame(
            [[0] * len(symptom_columns)],
            columns=symptom_columns
        )

        for symptom in selected_symptoms:
            input_data.loc[0, symptom] = 1

        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("3. Model Output")

        st.metric("Predicted dataset label", prediction)

        st.write("**Symptoms entered:**")
        st.write(", ".join(
            s.replace("_", " ").strip().title()
            for s in selected_symptoms
        ))

        st.info(
            "This is a dataset-based prediction, not a confirmed diagnosis. "
            "Symptoms can have many causes. Seek advice from a qualified "
            "healthcare professional for medical concerns."
        )

st.divider()
st.caption(
    "Academic prototype | Machine learning | Random Forest | Streamlit"
)
