from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import os
import io
import json
import joblib

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model_data = {}
file_data = {}

MODEL_CACHE_DIR = 'static/model_cache'
MODEL_CACHE_FILE = os.path.join(MODEL_CACHE_DIR, 'model_bundle.joblib')


def save_model_to_disk():
    """Persist the trained model bundle to disk so it survives restarts."""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    keys_to_save = ['model', 'scaler', 'le_target', 'class_names',
                    'numeric_cols', 'categorical_cols', 'x_columns']
    bundle = {k: model_data[k] for k in keys_to_save if k in model_data}
    joblib.dump(bundle, MODEL_CACHE_FILE)


def load_model_from_disk():
    """Load a previously saved model bundle from disk on startup."""
    if os.path.exists(MODEL_CACHE_FILE):
        try:
            bundle = joblib.load(MODEL_CACHE_FILE)
            model_data.update(bundle)
            return True
        except Exception:
            return False
    return False


# Auto-load saved model on startup
load_model_from_disk()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/guide')
def guide():
    return render_template('guide.html')


@app.route('/model_status', methods=['GET'])
def model_status():
    """Return whether a trained model is currently in memory."""
    if 'model' in model_data:
        return jsonify({
            'loaded': True,
            'class_names': model_data.get('class_names', []),
            'cached_on_disk': os.path.exists(MODEL_CACHE_FILE)
        })
    return jsonify({'loaded': False, 'cached_on_disk': os.path.exists(MODEL_CACHE_FILE)})


@app.route('/upload_labelled', methods=['POST'])
def upload_labelled():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.endswith('.csv'):
        try:
            file_content = file.read()
            df = pd.read_csv(io.BytesIO(file_content))
            file_data['labelled_data'] = df.copy()

            dataset_info = {
                'shape': df.shape,
                'missing_values': df.isnull().sum().to_dict(),
                'class_distribution': df['koi_disposition'].value_counts().to_dict(),
                'class_percentage': (df['koi_disposition'].value_counts(normalize=True) * 100).to_dict()
            }

            return jsonify({
                'success': True,
                'message': 'Labelled dataset uploaded successfully',
                'info': dataset_info
            })

        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file format'}), 400


@app.route('/load_demo_train', methods=['POST'])
def load_demo_train():
    try:
        demo_path = 'static/demo/train.csv'
        df = pd.read_csv(demo_path)
        file_data['labelled_data'] = df.copy()

        dataset_info = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'class_distribution': df['koi_disposition'].value_counts().to_dict(),
            'class_percentage': (df['koi_disposition'].value_counts(normalize=True) * 100).to_dict()
        }

        return jsonify({'success': True, 'message': 'Demo training dataset loaded', 'info': dataset_info})
    except Exception as e:
        return jsonify({'error': f'Error loading demo train: {str(e)}'}), 500


@app.route('/train_model', methods=['POST'])
def train_model():
    if 'labelled_data' not in file_data:
        return jsonify({'error': 'Please upload labelled dataset first'}), 400

    try:
        df = file_data['labelled_data'].copy()
        target_col = 'koi_disposition'
        df_clean = df.copy()

        y = df_clean[target_col].copy()
        x = df_clean.drop(columns=[target_col])

        numeric_cols = x.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = x.select_dtypes(include=['object']).columns.tolist()

        for col in numeric_cols:
            if x[col].isnull().sum() > 0:
                x.fillna({col: x[col].median()}, inplace=True)

        for col in categorical_cols:
            if x[col].isnull().sum() > 0:
                mode_val = x[col].mode()[0] if not x[col].mode().empty else 'UNKNOWN'
                x.fillna({col: mode_val}, inplace=True)

        if len(categorical_cols) > 0:
            for col in categorical_cols:
                le = LabelEncoder()
                x[col] = le.fit_transform(x[col].astype(str))

        for col in numeric_cols:
            Q1 = x[col].quantile(0.25)
            Q3 = x[col].quantile(0.75)
            IQR = Q3 - Q1
            x[col] = x[col].clip(Q1 - 3 * IQR, Q3 + 3 * IQR)

        if 'koi_prad' in x.columns and 'koi_srad' in x.columns:
            x['planets_to_star_radius_ratio'] = x['koi_prad'] / (x['koi_srad'] + 1e-10)
        if 'koi_period' in x.columns:
            x['log_period'] = np.log1p(x['koi_period'])
        if 'koi_depth' in x.columns and 'koi_duration' in x.columns:
            x['depth_to_duration'] = x['koi_depth'] / (x['koi_duration'] + 1e-10)

        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        class_names = le_target.classes_.tolist()

        scaler = StandardScaler()
        x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
        x_resampled, y_resampled = SMOTE(random_state=42).fit_resample(x_scaled, y_encoded)

        models = {
            'XGBoost': XGBClassifier(
                n_estimators=200, max_depth=20, learning_rate=0.1,
                random_state=42, n_jobs=-1, eval_metric='mlogloss'
            )
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = {}

        for model_name, model in models.items():
            cv_results = cross_validate(
                model, x_resampled, y_resampled, cv=cv,
                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                return_train_score=True, n_jobs=-1
            )
            model.fit(x_resampled, y_resampled)
            y_pred = model.predict(x_resampled)
            cm = confusion_matrix(y_resampled, y_pred)
            results[model_name] = {
                'cv_accuracy': cv_results['test_accuracy'].mean(),
                'cv_precision': cv_results['test_precision_macro'].mean(),
                'cv_recall': cv_results['test_recall_macro'].mean(),
                'cv_f1': cv_results['test_f1_macro'].mean(),
                'confusion_matrix': cm.tolist()
            }

        best_model_name = max(results.keys(), key=lambda k: results[k]['cv_accuracy'])
        best_model = models[best_model_name]

        model_data['model'] = best_model
        model_data['scaler'] = scaler
        model_data['le_target'] = le_target
        model_data['class_names'] = class_names
        model_data['numeric_cols'] = numeric_cols
        model_data['categorical_cols'] = categorical_cols
        model_data['x_columns'] = x.columns.tolist()

        # Persist trained model to disk
        save_model_to_disk()

        return jsonify({
            'success': True,
            'message': 'Model trained and saved successfully',
            'best_model': best_model_name,
            'results': results,
            'class_names': class_names
        })

    except Exception as e:
        return jsonify({'error': f'Error training model: {str(e)}'}), 500


@app.route('/upload_unlabelled', methods=['POST'])
def upload_unlabelled():
    if 'model' not in model_data:
        return jsonify({'error': 'Please train model first'}), 400

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and file.filename.endswith('.csv'):
        try:
            file_content = file.read()
            df_test = pd.read_csv(io.BytesIO(file_content))
            df_test_original = df_test.copy()
            x_test = df_test.copy()

            for col in model_data['numeric_cols']:
                if col in x_test.columns and x_test[col].isnull().sum() > 0:
                    x_test.fillna({col: x_test[col].median()}, inplace=True)

            for col in model_data['categorical_cols']:
                if col in x_test.columns and x_test[col].isnull().sum() > 0:
                    mode_val = x_test[col].mode()[0] if not x_test[col].mode().empty else 'UNKNOWN'
                    x_test.fillna({col: mode_val}, inplace=True)

            if 'koi_prad' in x_test.columns and 'koi_srad' in x_test.columns:
                x_test['planets_to_star_radius_ratio'] = x_test['koi_prad'] / (x_test['koi_srad'] + 1e-10)
            if 'koi_period' in x_test.columns:
                x_test['log_period'] = np.log1p(x_test['koi_period'])
            if 'koi_depth' in x_test.columns and 'koi_duration' in x_test.columns:
                x_test['depth_to_duration'] = x_test['koi_depth'] / (x_test['koi_duration'] + 1e-10)

            x_test = x_test[model_data['x_columns']]
            x_test_scaled = model_data['scaler'].transform(x_test)

            predictions = model_data['model'].predict(x_test_scaled)
            predictions_proba = model_data['model'].predict_proba(x_test_scaled)

            df_test_original['predicted_disposition'] = model_data['le_target'].inverse_transform(predictions)
            model_data['test_data'] = df_test_original
            model_data['predictions'] = predictions
            model_data['predictions_proba'] = predictions_proba

            return jsonify({
                'success': True,
                'message': 'Predictions completed',
                'total_rows': len(df_test_original),
                'prediction_summary': df_test_original['predicted_disposition'].value_counts().to_dict()
            })

        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file format'}), 400


@app.route('/load_demo_test', methods=['POST'])
def load_demo_test():
    if 'model' not in model_data:
        return jsonify({'error': 'Please train model first'}), 400
    try:
        demo_path = 'static/demo/test.csv'
        df_test = pd.read_csv(demo_path)
        df_test_original = df_test.copy()
        x_test = df_test.copy()

        for col in model_data.get('numeric_cols', []):
            if col in x_test.columns and x_test[col].isnull().sum() > 0:
                x_test.fillna({col: x_test[col].median()}, inplace=True)

        for col in model_data.get('categorical_cols', []):
            if col in x_test.columns and x_test[col].isnull().sum() > 0:
                mode_val = x_test[col].mode()[0] if not x_test[col].mode().empty else 'UNKNOWN'
                x_test.fillna({col: mode_val}, inplace=True)

        if 'koi_prad' in x_test.columns and 'koi_srad' in x_test.columns:
            x_test['planets_to_star_radius_ratio'] = x_test['koi_prad'] / (x_test['koi_srad'] + 1e-10)
        if 'koi_period' in x_test.columns:
            x_test['log_period'] = np.log1p(x_test['koi_period'])
        if 'koi_depth' in x_test.columns and 'koi_duration' in x_test.columns:
            x_test['depth_to_duration'] = x_test['koi_depth'] / (x_test['koi_duration'] + 1e-10)

        x_test = x_test.reindex(columns=model_data['x_columns'], fill_value=0)
        x_test_scaled = model_data['scaler'].transform(x_test)

        predictions = model_data['model'].predict(x_test_scaled)
        predictions_proba = model_data['model'].predict_proba(x_test_scaled)

        df_test_original['predicted_disposition'] = model_data['le_target'].inverse_transform(predictions)
        model_data['test_data'] = df_test_original
        model_data['predictions'] = predictions
        model_data['predictions_proba'] = predictions_proba

        return jsonify({
            'success': True,
            'message': 'Demo test dataset predicted',
            'total_rows': len(df_test_original),
            'prediction_summary': df_test_original['predicted_disposition'].value_counts().to_dict()
        })
    except Exception as e:
        return jsonify({'error': f'Error loading demo test: {str(e)}'}), 500


@app.route('/predict_row/<int:row_num>', methods=['GET'])
def predict_row(row_num):
    if 'test_data' not in model_data:
        return jsonify({'error': 'No test data available'}), 400

    try:
        idx = row_num - 1
        if idx < 0 or idx >= len(model_data['test_data']):
            return jsonify({'error': 'Invalid row number'}), 400

        predicted_class = model_data['predictions'][idx]
        predicted_label = model_data['le_target'].inverse_transform([predicted_class])[0]
        probabilities = model_data['predictions_proba'][idx]
        prob_dict = {name: float(probabilities[i]) for i, name in enumerate(model_data['class_names'])}
        sample_data = model_data['test_data'].iloc[idx].to_dict()

        return jsonify({
            'success': True,
            'row_number': row_num,
            'predicted_label': predicted_label,
            'probabilities': prob_dict,
            'features': sample_data
        })

    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


@app.route('/clear_data', methods=['POST'])
def clear_data():
    file_data.clear()
    model_data.clear()
    # Also remove the cached model from disk
    if os.path.exists(MODEL_CACHE_FILE):
        os.remove(MODEL_CACHE_FILE)
    return jsonify({'success': True, 'message': 'All data cleared from memory and disk'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
