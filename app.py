import streamlit as st
from data_fetcher import FootballDataAPI, prepare_match_data, clean_team_name
from predictor import MatchPredictor
import pandas as pd

def main():
    st.title("Premier League Match Predictor")
    
    # Get API key from secrets
    API_KEY = st.secrets["FOOTBALL_API_KEY"]
    
    # Initialize API and load data
    api = FootballDataAPI(API_KEY)
    
    # Load and train model
    with st.spinner("Loading historical data and training model..."):
        match_data = prepare_match_data(api)
        predictor = MatchPredictor()
        training_results = predictor.train(match_data)
        
        # Display model performance metrics
        st.success(f"Model trained with accuracy: {training_results['accuracy']:.2f}")
        st.write(f"Precision: {training_results['precision']:.2f}")
        st.write(f"Recall: {training_results['recall']:.2f}")
        st.write(f"F1 Score: {training_results['f1']:.2f}")
        
        # Display feature importance
        st.subheader("Feature Importance")
        feature_importance = training_results['feature_importance']
        st.bar_chart(data=feature_importance.set_index('feature')['importance'])
        
        # Display classification report
        st.subheader("Model Performance Details")
        st.text(training_results['classification_report'])
    
    # Get upcoming matches
    upcoming = api.get_upcoming_matches()
    
    if 'matches' in upcoming:
        st.header("Upcoming Match Predictions")
        
        for match in upcoming['matches']:
            home_team = clean_team_name(match['homeTeam']['name'])
            away_team = clean_team_name(match['awayTeam']['name'])
            match_date = match['utcDate']
            
            # Make prediction
            prediction = predictor.predict_match(home_team, away_team)
            
            # Create a container for each match
            with st.container():
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**{home_team}**")
                with col2:
                    st.write("vs")
                with col3:
                    st.write(f"**{away_team}**")
                
                # Display prediction and probabilities
                result = "Home Win" if prediction['prediction'] == 1 else "Away Win" if prediction['prediction'] == -1 else "Draw"
                
                # Create probability bars
                probs = prediction['probabilities']
                st.write("Prediction Probabilities:")
                st.progress(probs['home_win'])
                st.write(f"Home Win: {probs['home_win']:.2%}")
                st.progress(probs['draw'])
                st.write(f"Draw: {probs['draw']:.2%}")
                st.progress(probs['away_win'])
                st.write(f"Away Win: {probs['away_win']:.2%}")
                
                st.write(f"**Final Prediction:** {result}")
                st.write(f"Match Date: {match_date}")
                st.divider()

if __name__ == "__main__":
    main() 