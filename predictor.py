from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

def clean_team_name(name):
    """Clean and standardize team names"""
    replacements = {
        'AFC ': '',
        ' FC': '',
        'Hotspur': '',
        '& ': 'and ',
        'United': 'Utd',
        'Manchester': 'Man',
        'Wolverhampton': 'Wolves',
        'Wanderers': '',
        'Forest': 'For',
        'Brighton and Hove Albion': 'Brighton',
        'Newcastle Utd': 'Newcastle',
        'Leeds Utd': 'Leeds',
        'Leicester City': 'Leicester',
        'Manchester City': 'Man City',
        'Manchester United': 'Man Utd',
        'Nottingham Forest': 'Nottingham For',
        'Tottenham': 'Spurs'
    }
    
    name = name.strip()
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name.strip()

def safe_ratio(a, b, default=0.0):
    """Safely calculate a ratio with clipping"""
    if b == 0 or b is None or np.isnan(b):
        return default
    ratio = a / b
    return np.clip(ratio, -5, 5)

class MatchPredictor:
    def __init__(self):
        self.le_teams = LabelEncoder()
        self.scaler = StandardScaler()
        self.le_results = LabelEncoder()
        
        # Enhanced model ensemble
        self.rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42
        )
        
        self.gb = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        self.xgb = XGBClassifier(
            n_estimators=500,
            max_depth=15,
            learning_rate=0.05,
            subsample=0.8,
            scale_pos_weight=1,
            random_state=42
        )
        
        # Add Neural Network to ensemble
        self.nn = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
        
        self.models = [self.rf, self.gb, self.xgb, self.nn]
        self.model_weights = [0.3, 0.25, 0.25, 0.2]  # Adjusted weights for ensemble
        self.trained_models = []
        
    def prepare_features(self, df):
        """Enhanced feature preparation with advanced metrics"""
        # Define the exact set of features we'll use
        self.feature_list = [
            'home_team_encoded', 'away_team_encoded',
            'home_position', 'away_position',
            'position_difference',
            'home_points', 'away_points',
            'home_goal_diff', 'away_goal_diff',
            'home_form', 'away_form',
            'home_win_rate', 'away_win_rate',
            'home_clean_sheet_rate', 'away_clean_sheet_rate',
            'home_avg_goals_scored', 'away_avg_goals_scored',
            'home_avg_goals_conceded', 'away_avg_goals_conceded',
            'ht_result',
            'goal_difference',
            'form_momentum'
        ]
        
        # Basic encoding
        df['home_team_encoded'] = self.le_teams.fit_transform(df['home_team'])
        df['away_team_encoded'] = self.le_teams.transform(df['away_team'])
        
        # Create target variable
        df['result'] = np.where(df['home_score'] > df['away_score'], 1,
                              np.where(df['home_score'] < df['away_score'], -1, 0))
        
        # Transform results for XGBoost compatibility
        y = self.le_results.fit_transform(df['result'])
        
        # Calculate additional features
        df['goal_difference'] = np.clip(
            df['home_avg_goals_scored'] - df['away_avg_goals_scored'],
            -5, 5
        )
        df['form_momentum'] = np.clip(df['home_form'] - df['away_form'], -1, 1)
        
        X = df[self.feature_list].copy()
        
        # Replace infinities and NaNs before scaling
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaNs with appropriate values
        for col in X.columns:
            if col in ['home_team_encoded', 'away_team_encoded', 'ht_result']:
                continue
            elif 'rate' in col or 'efficiency' in col or 'form' in col:
                X[col] = X[col].fillna(0.5)
            elif 'position' in col:
                X[col] = X[col].fillna(10)
            else:
                X[col] = X[col].fillna(X[col].mean())
        
        # Scale numerical features
        numerical_features = [f for f in self.feature_list 
                             if f not in ['home_team_encoded', 'away_team_encoded', 'ht_result']]
        X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        return X, y
    
    def train(self, match_data):
        X, y = self.prepare_features(match_data)
        
        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        predictions = []
        probabilities = []
        feature_importances = []
        
        # Train models with progress tracking
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}...")
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)
            predictions.append(pred)
            probabilities.append(prob)
            
            # Only collect feature importance from models that support it
            if hasattr(model, 'feature_importances_'):
                feature_importances.append(model.feature_importances_)
        
        self.trained_models = self.models
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros_like(predictions[0], dtype=float)
        ensemble_prob = np.zeros_like(probabilities[0])
        
        for pred, prob, weight in zip(predictions, probabilities, self.model_weights):
            ensemble_pred += weight * pred
            ensemble_prob += weight * prob
        
        # Convert predictions back to original labels
        ensemble_pred = self.le_results.inverse_transform(ensemble_pred.round().astype(int))
        y_test_original = self.le_results.inverse_transform(y_test)
        
        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_original, ensemble_pred, average='weighted')
        
        # Calculate average feature importance only for models that support it
        if feature_importances:
            # Get valid weights and normalize them
            valid_weights = []
            valid_importances = []
            
            for i, model in enumerate(self.models):
                if hasattr(model, 'feature_importances_'):
                    valid_weights.append(self.model_weights[i])
                    valid_importances.append(model.feature_importances_)
            
            # Normalize weights
            valid_weights = np.array(valid_weights)
            valid_weights = valid_weights / valid_weights.sum()
            
            # Calculate weighted average importance
            avg_importance = np.zeros_like(valid_importances[0])
            for imp, weight in zip(valid_importances, valid_weights):
                avg_importance += weight * imp
        else:
            avg_importance = np.zeros(len(X.columns))
        
        return {
            'accuracy': accuracy_score(y_test_original, ensemble_pred),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'feature_importance': pd.DataFrame({
                'feature': X.columns,
                'importance': avg_importance
            }).sort_values('importance', ascending=False),
            'classification_report': classification_report(y_test_original, ensemble_pred)
        }
    
    def predict_match(self, home_team, away_team, home_stats=None, away_stats=None):
        # Default statistics if none provided
        default_stats = {
            'position': 10,
            'points': 20,
            'goal_diff': 0,
            'form': 0.5,
            'win_rate': 0.5,
            'clean_sheet_rate': 0.3,
            'avg_goals_scored': 1.5,
            'avg_goals_conceded': 1.5,
            'ht_result': 0
        }
        
        home_stats = home_stats or default_stats.copy()
        away_stats = away_stats or default_stats.copy()
        
        # Encode teams
        home_encoded = self.le_teams.transform([home_team])[0]
        away_encoded = self.le_teams.transform([away_team])[0]
        
        # Calculate additional metrics
        goal_difference = np.clip(
            home_stats['avg_goals_scored'] - away_stats['avg_goals_scored'],
            -5, 5
        )
        form_momentum = np.clip(
            home_stats['form'] - away_stats['form'],
            -1, 1
        )
        
        # Create feature array matching training features exactly
        features = np.array([[
            home_encoded, away_encoded,
            home_stats['position'], away_stats['position'],
            home_stats['position'] - away_stats['position'],
            home_stats['points'], away_stats['points'],
            home_stats['goal_diff'], away_stats['goal_diff'],
            home_stats['form'], away_stats['form'],
            home_stats['win_rate'], away_stats['win_rate'],
            home_stats['clean_sheet_rate'], away_stats['clean_sheet_rate'],
            home_stats['avg_goals_scored'], away_stats['avg_goals_scored'],
            home_stats['avg_goals_conceded'], away_stats['avg_goals_conceded'],
            home_stats['ht_result'],
            goal_difference,
            form_momentum
        ]])
        
        # Scale numerical features
        numerical_features_mask = [i for i, f in enumerate(self.feature_list) 
                                 if f not in ['home_team_encoded', 'away_team_encoded', 'ht_result']]
        features[:, numerical_features_mask] = self.scaler.transform(features[:, numerical_features_mask])
        
        # Get predictions from all models
        predictions = []
        probabilities = []
        
        for model in self.trained_models:
            pred = model.predict(features)
            prob = model.predict_proba(features)
            predictions.append(pred)
            probabilities.append(prob)
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros_like(predictions[0], dtype=float)
        ensemble_prob = np.zeros_like(probabilities[0])
        
        for pred, prob, weight in zip(predictions, probabilities, self.model_weights):
            ensemble_pred += weight * pred
            ensemble_prob += weight * prob
        
        # Convert prediction back to original labels
        final_prediction = self.le_results.inverse_transform([round(ensemble_pred[0])])[0]
        
        # Map probabilities to outcomes
        probs = {
            'home_win': float(ensemble_prob[0][2]),  # Class index 2 for home win
            'draw': float(ensemble_prob[0][1]),      # Class index 1 for draw
            'away_win': float(ensemble_prob[0][0])   # Class index 0 for away win
        }
        
        return {
            'prediction': final_prediction,
            'probabilities': probs
        }