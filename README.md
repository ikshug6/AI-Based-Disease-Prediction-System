# AI-Based Disease Prediction System

## Project Overview
This project is an educational machine learning prototype that predicts a disease label based on symptoms selected by the user. It uses a Random Forest classifier and a Streamlit web interface.

## Objectives
- Prepare and clean a symptom-based disease dataset.
- Train a machine learning model for disease prediction.
- Evaluate the model using the supplied testing dataset.
- Provide a simple interface for selecting symptoms and viewing predictions.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn

## System Workflow
Dataset → Data Cleaning → Feature Preparation → Random Forest Training → Model Evaluation → Streamlit Interface → Predicted Disease Label

## Project Files
- `app.py` — Streamlit application.
- `train_model.py` — model training script.
- `clean_data.py` — dataset cleaning script.
- `evaluate_model.py` and other evaluation scripts — model evaluation and data analysis.
- `disease_model.pkl` — saved trained model.
- `symptom_columns.pkl` — saved symptom-column information.
- `Training_clean.csv` and `Testing_clean.csv` — cleaned datasets.
- `screenshots/` — project screenshots, if available.

## Dataset
Source: Kaggle — Disease Prediction Using Machine Learning  
https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning

Please check the dataset's license and redistribution conditions before publishing dataset files.

## How to Run
1. Install Python 3.
2. Open Terminal in the project folder.
3. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

## Evaluation and Limitations
The Random Forest model achieved 100% accuracy on the supplied 42-record test dataset in the current evaluation. However, 41 of the 42 test symptom patterns also appeared in the training data. Therefore, this result may overestimate performance on unseen cases and should be interpreted cautiously.

This prototype is for educational purposes only. It is not a clinically validated diagnostic tool and must not be used to make medical decisions.

## Future Improvements
- Evaluate using more independent, unseen data.
- Improve validation and performance reporting.
- Expand testing and improve the user interface.
