import streamlit as st
from data_fetcher import FootballDataAPI, prepare_match_data, clean_team_name
from predictor import MatchPredictor
import pandas as pd
from datetime import datetime

def main():
    st.title("Premier League Match Predictor")
    
    # Get API key from secrets
    API_KEY = st.secrets["FOOTBALL_API_KEY"]
    
    # Initialize API and predictor
    api = FootballDataAPI(API_KEY)
    predictor = MatchPredictor()
    
    # Try to load existing model
    last_updated = predictor.load_model()
    
    # Check if we need to train a new model
    if predictor.needs_update(last_updated):
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Data loading phase
        status_text.text("Loading historical match data...")
        match_data = prepare_match_data(api, progress_callback=lambda x: progress_bar.progress(x * 0.5))
        
        # Model training phase
        status_text.text("Training prediction models...")
        training_results = predictor.train(match_data, 
                                         progress_callback=lambda x: progress_bar.progress(0.5 + x * 0.5))
        
        # Save the model
        status_text.text("Saving model...")
        predictor.save_model()
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show success message and metrics
        st.success("Model training completed!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Accuracy", value=f"{training_results['accuracy']:.2%}")
        with col2:
            st.metric(label="Precision", value=f"{training_results['precision']:.2%}")
        with col3:
            st.metric(label="Recall", value=f"{training_results['recall']:.2%}")
        with col4:
            st.metric(label="F1 Score", value=f"{training_results['f1']:.2%}")
        
        # Display feature importance
        st.subheader("Feature Importance")
        feature_importance = training_results['feature_importance']
        st.bar_chart(data=feature_importance.set_index('feature')['importance'])
        
        # Display classification report
        with st.expander("Detailed Performance Metrics"):
            st.text(training_results['classification_report'])
    else:
        st.info(f"Using existing model (last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # Get upcoming matches
    with st.spinner("Fetching upcoming matches..."):
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
                # Match header
                col1, col2, col3 = st.columns([2,1,2])
                with col1:
                    st.markdown(f"<h3 style='text-align: right'>{home_team}</h3>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<h3 style='text-align: center'>vs</h3>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<h3 style='text-align: left'>{away_team}</h3>", unsafe_allow_html=True)
                
                # Display prediction and probabilities
                result = "Home Win" if prediction['prediction'] == 1 else "Away Win" if prediction['prediction'] == -1 else "Draw"
                probs = prediction['probabilities']
                
                # Create columns for probabilities
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                
                with prob_col1:
                    st.metric("Home Win", f"{probs['home_win']:.1%}")
                    st.progress(probs['home_win'])
                
                with prob_col2:
                    st.metric("Draw", f"{probs['draw']:.1%}")
                    st.progress(probs['draw'])
                
                with prob_col3:
                    st.metric("Away Win", f"{probs['away_win']:.1%}")
                    st.progress(probs['away_win'])
                
                # Display final prediction and match date
                st.markdown(f"**Prediction:** {result}")
                st.markdown(f"**Match Date:** {datetime.strptime(match_date, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d %H:%M')}")
                st.divider()
    else:
        st.info("No upcoming matches found for the next week.")

if __name__ == "__main__":
    main() 