import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'Crop_recommendation.csv'
OUTPUT_DIR = BASE_DIR / 'crop_recommendation_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


def save_plot(filename: str):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing expected columns: {missing}')
    return df


def perform_eda(df: pd.DataFrame):
    print('\n=== BASIC DATA AUDIT ===')
    print(df.head())
    print('\nShape:', df.shape)
    print('\nMissing values:\n', df.isna().sum())
    print('\nData types:\n', df.dtypes)
    print('\nClass counts:\n', df['label'].value_counts().sort_index())

    summary = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'n_crop_classes': int(df['label'].nunique()),
        'missing_values_total': int(df.isna().sum().sum()),
        'class_balance_equal': bool(df['label'].value_counts().nunique() == 1),
    }
    with open(OUTPUT_DIR / 'data_audit_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Class distribution
    plt.figure(figsize=(12, 6))
    df['label'].value_counts().sort_values(ascending=False).plot(kind='bar')
    plt.title('Crop Class Distribution')
    plt.xlabel('Crop label')
    plt.ylabel('Count')
    plt.xticks(rotation=60, ha='right')
    save_plot('01_class_distribution.png')

    # Correlation heatmap
    corr = df.drop(columns='label').corr()
    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr, aspect='auto')
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Feature Correlation Heatmap')
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
    save_plot('02_correlation_heatmap.png')

    # Crop-wise mean feature heatmap
    mean_profile = df.groupby('label')[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()
    plt.figure(figsize=(10, 9))
    im = plt.imshow(mean_profile, aspect='auto')
    plt.colorbar(im)
    plt.xticks(range(len(mean_profile.columns)), mean_profile.columns, rotation=45, ha='right')
    plt.yticks(range(len(mean_profile.index)), mean_profile.index)
    plt.title('Crop-wise Mean Environmental/Nutrient Profile')
    save_plot('03_crop_feature_profile_heatmap.png')

    # Boxplots for key variables
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    for feature in features:
        plt.figure(figsize=(12, 6))
        df.boxplot(column=feature, by='label', rot=60)
        plt.title(f'{feature} by Crop')
        plt.suptitle('')
        plt.xlabel('Crop label')
        plt.ylabel(feature)
        save_plot(f'boxplot_{feature}.png')


def split_data(df: pd.DataFrame):
    X = df.drop(columns='label')
    y = df['label']
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


def tune_models(X_train: pd.DataFrame, y_train: pd.Series):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    rf_grid = {
        'n_estimators': [150, 250],
        'max_depth': [None, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }
    rf_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid=rf_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    rf_search.fit(X_train, y_train)

    ada_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.5, 1.0],
        'estimator': [
            DecisionTreeClassifier(max_depth=1, random_state=42),
            DecisionTreeClassifier(max_depth=2, random_state=42)
        ]
    }
    ada_search = GridSearchCV(
        estimator=AdaBoostClassifier(random_state=42),
        param_grid=ada_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    ada_search.fit(X_train, y_train)

    return rf_search, ada_search


def evaluate_model(model, model_name: str, X_test: pd.DataFrame, y_test: pd.Series):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, pred)),
        'macro_f1': float(f1_score(y_test, pred, average='macro')),
        'macro_precision': float(precision_score(y_test, pred, average='macro')),
        'macro_recall': float(recall_score(y_test, pred, average='macro')),
        'top_3_accuracy': float(top_k_accuracy_score(y_test, proba, k=3, labels=model.classes_)),
    }

    print(f'\n=== {model_name.upper()} RESULTS ===')
    print(pd.Series(metrics))
    report = classification_report(y_test, pred, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / f'{model_name}_classification_report.csv', index=True)

    cm = confusion_matrix(y_test, pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    save_plot(f'{model_name}_confusion_matrix.png')

    return metrics, pred, proba


def plot_model_comparison(metrics_dict: dict):
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.to_csv(OUTPUT_DIR / 'model_comparison_metrics.csv', index=True)

    plt.figure(figsize=(10, 6))
    metrics_df[['accuracy', 'macro_f1', 'top_3_accuracy']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.ylim(0, 1.05)
    save_plot('model_performance_comparison.png')


def plot_feature_importance(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str):
    # Native feature importance
    if hasattr(model, 'feature_importances_'):
        native_imp = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=True)
        native_imp.to_csv(OUTPUT_DIR / f'{model_name}_native_feature_importance.csv', header=['importance'])
        plt.figure(figsize=(8, 5))
        native_imp.plot(kind='barh')
        plt.title(f'{model_name} Native Feature Importance')
        plt.xlabel('Importance')
        save_plot(f'{model_name}_native_feature_importance.png')

    # Permutation importance (stronger interpretation)
    perm = permutation_importance(model, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=True)
    perm_imp.to_csv(OUTPUT_DIR / f'{model_name}_permutation_importance.csv', header=['importance'])
    plt.figure(figsize=(8, 5))
    perm_imp.plot(kind='barh')
    plt.title(f'{model_name} Permutation Importance')
    plt.xlabel('Mean accuracy decrease')
    save_plot(f'{model_name}_permutation_importance.png')


def shap_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str):
    try:
        import shap

        sample_train = X_train.sample(min(150, len(X_train)), random_state=42)
        sample_test = X_test.sample(min(120, len(X_test)), random_state=42)

        if model_name.lower() == 'random_forest':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_test)
        else:
            # AdaBoost multiclass SHAP can be heavy; use model-agnostic explainer on smaller sample.
            background = sample_train.sample(min(40, len(sample_train)), random_state=42)
            explainer = shap.Explainer(model.predict_proba, background)
            shap_values = explainer(sample_test)

        # Summary bar plot
        plt.figure()
        try:
            shap.summary_plot(shap_values, sample_test, plot_type='bar', show=False)
            plt.title(f'{model_name} SHAP Summary (Bar)')
            save_plot(f'{model_name}_shap_summary_bar.png')
        except Exception:
            pass

    except Exception as e:
        print(f'SHAP skipped for {model_name}: {e}')


def lime_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str):
    try:
        from lime.lime_tabular import LimeTabularExplainer

        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=list(X_train.columns),
            class_names=list(model.classes_),
            mode='classification'
        )
        sample_row = X_test.iloc[0].values
        explanation = explainer.explain_instance(
            data_row=sample_row,
            predict_fn=model.predict_proba,
            num_features=len(X_train.columns),
            top_labels=1,
        )
        html_path = OUTPUT_DIR / f'{model_name}_lime_explanation.html'
        explanation.save_to_file(str(html_path))
        print(f'LIME explanation saved to: {html_path}')
    except Exception as e:
        print(
            f'LIME skipped for {model_name}: {e}. '\
            'Install with: pip install lime'
        )


def save_artifacts(rf_model, ada_model, feature_ranges: dict):
    joblib.dump(rf_model, OUTPUT_DIR / 'random_forest_crop_model.joblib')
    joblib.dump(ada_model, OUTPUT_DIR / 'adaboost_crop_model.joblib')
    with open(OUTPUT_DIR / 'feature_ranges.json', 'w', encoding='utf-8') as f:
        json.dump(feature_ranges, f, indent=2)


def main():
    df = load_data(DATA_PATH)
    perform_eda(df)

    X_train, X_test, y_train, y_test = split_data(df)
    feature_ranges = {
        col: {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean())
        }
        for col in X_train.columns
    }

    rf_search, ada_search = tune_models(X_train, y_train)
    rf_model = rf_search.best_estimator_
    ada_model = ada_search.best_estimator_

    print('\nBest RF params:', rf_search.best_params_)
    print('Best AdaBoost params:', ada_search.best_params_)

    rf_metrics, _, _ = evaluate_model(rf_model, 'random_forest', X_test, y_test)
    ada_metrics, _, _ = evaluate_model(ada_model, 'adaboost', X_test, y_test)

    plot_model_comparison({
        'Random Forest': rf_metrics,
        'AdaBoost': ada_metrics,
    })

    plot_feature_importance(rf_model, X_test, y_test, 'random_forest')
    plot_feature_importance(ada_model, X_test, y_test, 'adaboost')

    shap_analysis(rf_model, X_train, X_test, 'random_forest')
    shap_analysis(ada_model, X_train, X_test, 'adaboost')

    lime_analysis(rf_model, X_train, X_test, 'random_forest')
    lime_analysis(ada_model, X_train, X_test, 'adaboost')

    save_artifacts(rf_model, ada_model, feature_ranges)
    print('\nAll outputs saved to:', OUTPUT_DIR)


if __name__ == '__main__':
    main()
