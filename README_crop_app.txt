Crop Recommendation Workflow + App

Files included:
1. crop_recommendation_rf_adaboost_workflow.py  -> training, EDA, grid search, evaluation, feature importance, SHAP, LIME hooks
2. app.py                                       -> Streamlit prediction app for farmers
3. Crop_recommendation.csv                      -> source dataset

How to run locally:
1. Open terminal or Anaconda Prompt
2. Change directory to the folder containing these files
3. Install packages:
   pip install pandas numpy matplotlib scikit-learn streamlit shap joblib lime
4. Train models and generate outputs:
   python crop_recommendation_rf_adaboost_workflow.py
5. Launch app:
   streamlit run app.py

Important note:
- If LIME is not installed, the workflow will skip LIME gracefully and tell you the install command.
- The app expects the training script to run first because it loads saved model files from crop_recommendation_outputs.
