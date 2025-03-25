import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import lightgbm as lgb
from dataProcessing import BasketballDataProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="March Madness 2025 Predictor",
    page_icon="🏀",
    layout="wide"
)

# App title and description
st.title("March Madness 2025 Predictor")
st.markdown("Generate predictions for all possible matchups in the NCAA Men's and Women's Basketball Tournament 2025")

# Initialize session state for storing data between reruns
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.predictions_df = None

# Function to load team data
@st.cache_data
def load_team_data():
    mens_teams = pd.read_csv("data/MTeams.csv")
    womens_teams = pd.read_csv("data/WTeams.csv")
    return mens_teams, womens_teams

# Function to load trained models
@st.cache_resource
def load_models():
    # Load men's model
    mens_model = lgb.Booster(model_file="models/mens_model.txt")
    with open("models/preprocessing_pipeline_mens.pkl", "rb") as f:
        mens_pipeline = pickle.load(f)
    
    # Load women's model
    womens_model = lgb.Booster(model_file="models/womens_model.txt")
    with open("models/preprocessing_pipeline_womens.pkl", "rb") as f:
        womens_pipeline = pickle.load(f)
    
    return mens_model, mens_pipeline, womens_model, womens_pipeline

# Load data and models
mens_teams, womens_teams = load_team_data()
try:
    mens_model, mens_pipeline, womens_model, womens_pipeline = load_models()
    if not st.session_state.data_loaded:
        st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")

# Function to process 2025 season data
@st.cache_data
def process_season_data():
    # Process men's data
    mens_processor = BasketballDataProcessor("data", is_mens=True)
    mens_processor.load_data()
    mens_processor.precompute_season_stats(2025)
    mens_processor.precompute_detailed_stats(2025)
    
    # Process women's data
    womens_processor = BasketballDataProcessor("data", is_mens=False)
    womens_processor.load_data()
    womens_processor.precompute_season_stats(2025)
    womens_processor.precompute_detailed_stats(2025)
    
    return mens_processor, womens_processor

# Process 2025 data - only show spinner if not loaded before
if not st.session_state.data_loaded:
    with st.spinner("Processing 2025 season data..."):
        mens_processor, womens_processor = process_season_data()
        st.success("Season data processed successfully!")
else:
    mens_processor, womens_processor = process_season_data()

# Load tournament seeds and slots
@st.cache_data
def load_tournament_data():
    # Men's data
    mens_seeds = pd.read_csv("data/MNCAATourneySeeds.csv")
    mens_seeds_2025 = mens_seeds[mens_seeds['Season'] == 2025]
    mens_slots = pd.read_csv("data/MNCAATourneySlots.csv")
    mens_slots_2025 = mens_slots[mens_slots['Season'] == 2025]
    
    # Women's data
    womens_seeds = pd.read_csv("data/WNCAATourneySeeds.csv")
    womens_seeds_2025 = womens_seeds[womens_seeds['Season'] == 2025]
    womens_slots = pd.read_csv("data/WNCAATourneySlots.csv")
    womens_slots_2025 = womens_slots[womens_slots['Season'] == 2025]
    
    return mens_seeds_2025, mens_slots_2025, womens_seeds_2025, womens_slots_2025

# Load tournament data
mens_seeds_2025, mens_slots_2025, womens_seeds_2025, womens_slots_2025 = load_tournament_data()

# Function to create features for a matchup
def create_matchup_features(team1_id, team2_id, processor):
    features = processor.create_matchup_features(team1_id, team2_id, 2025)
    if features is None:
        return None
    
    # Remove non-feature columns
    non_feature_cols = ['Season', 'Team1ID', 'Team2ID']
    features = {k: v for k, v in features.items() if k not in non_feature_cols}
    return pd.DataFrame([features])

# Function to predict matchup outcome
def predict_matchup(team1_id, team2_id, processor, model, pipeline, is_mens=True):
    features_df = create_matchup_features(team1_id, team2_id, processor)
    if features_df is None:
        return 0.5  # Default probability if we can't get features
    
    # Process features
    X = pipeline.transform(features_df)
    
    # Make prediction
    pred = model.predict(X)[0]
    
    # Convert to probability
    return 1 / (1 + np.exp(-pred))

# Function to map team ID to team name
def get_team_name(team_id, teams_df):
    team = teams_df[teams_df['TeamID'] == team_id]
    if len(team) > 0:
        return team.iloc[0]['TeamName']
    return f"Team {team_id}"

# Function to predict all matchups from SampleSubmissionStage2
def predict_all_matchups():
    # Load sample submission
    sample_df = pd.read_csv("data`/SampleSubmissionStage2.csv")
    
    # Create predictions
    predictions = []
    
    # Setup progress tracking
    total = len(sample_df)
    
    # Process each matchup
    for i, row in enumerate(sample_df.iterrows()):
        id_str = row[1]['ID']
        parts = id_str.split('_')
        
        # Parse ID format: 2025_team1_team2
        team1_id = int(parts[1])
        team2_id = int(parts[2])
        
        # Determine if men's or women's game (men's IDs < 3000)
        is_mens = team1_id < 3000
        
        # Get team names
        teams_df = mens_teams if is_mens else womens_teams
        team1_name = get_team_name(team1_id, teams_df)
        team2_name = get_team_name(team2_id, teams_df)
        
        # Select appropriate processor and model
        processor = mens_processor if is_mens else womens_processor
        model = mens_model if is_mens else womens_model
        pipeline = mens_pipeline if is_mens else womens_pipeline
        
        # Predict outcome
        prob = predict_matchup(team1_id, team2_id, processor, model, pipeline, is_mens)
        
        # Store prediction
        predictions.append({
            'ID': id_str,
            'Team1': team1_name,
            'Team2': team2_name,
            'Team1_Win_Probability': prob if team1_id < team2_id else (1 - prob),
            'Pred': prob  # Keep original column for submission
        })
        
        # Report progress every 1000 items 
        if i % 1000 == 0:
            st.write(f"Processed {i} of {total} matchups ({i/total:.1%})")
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame(predictions)
    
    # Save to CSV (with only the required columns for submission)
    submission_df = predictions_df[['ID', 'Pred']]
    submission_df.to_csv("submission.csv", index=False)
    
    return predictions_df

# Function to load existing submission file and enhance it with team names
def load_existing_submission():
    if not os.path.exists("submission.csv"):
        return None
    
    # Load submission file
    submission_df = pd.read_csv("submission.csv")
    
    # Enhance with team names
    predictions = []
    for _, row in submission_df.iterrows():
        id_str = row['ID']
        parts = id_str.split('_')
        
        # Parse ID format: 2025_team1_team2
        team1_id = int(parts[1])
        team2_id = int(parts[2])
        
        # Determine if men's or women's game
        is_mens = team1_id < 3000
        
        # Get team names
        teams_df = mens_teams if is_mens else womens_teams
        team1_name = get_team_name(team1_id, teams_df)
        team2_name = get_team_name(team2_id, teams_df)
        
        predictions.append({
            'ID': id_str,
            'Team1': team1_name,
            'Team2': team2_name,
            'Team1_Win_Probability': row['Pred'] if team1_id < team2_id else (1 - row['Pred']),
            'Pred': row['Pred']
        })
    
    return pd.DataFrame(predictions)

# Function to determine Sweet 16 teams based on predictions
def get_sweet_16_teams(predictions_df, is_mens=True):
    # Filter for men's or women's teams
    if is_mens:
        filtered_df = predictions_df[predictions_df['ID'].str.contains('_1')]  # Men's IDs < 3000
    else:
        filtered_df = predictions_df[predictions_df['ID'].str.contains('_3')]  # Women's IDs >= 3000
    
    # Calculate average win probability for each team
    team_stats = {}
    
    # Go through each matchup
    for _, row in filtered_df.iterrows():
        parts = row['ID'].split('_')
        team1_id = int(parts[1])
        team2_id = int(parts[2])
        prob = row['Pred']
        
        # Add Team1 stats
        if team1_id not in team_stats:
            team_stats[team1_id] = {'wins': 0, 'total': 0, 'avg_prob': 0}
        team_stats[team1_id]['wins'] += prob
        team_stats[team1_id]['total'] += 1
        
        # Add Team2 stats
        if team2_id not in team_stats:
            team_stats[team2_id] = {'wins': 0, 'total': 0, 'avg_prob': 0}
        team_stats[team2_id]['wins'] += (1 - prob)
        team_stats[team2_id]['total'] += 1
    
    # Calculate average win probability
    for team_id in team_stats:
        if team_stats[team_id]['total'] > 0:
            team_stats[team_id]['avg_prob'] = team_stats[team_id]['wins'] / team_stats[team_id]['total']
    
    # Sort teams by average win probability
    sorted_teams = sorted(team_stats.items(), key=lambda x: x[1]['avg_prob'], reverse=True)
    
    # Get top 16 teams
    sweet_16 = sorted_teams[:16]
    
    # Get team details
    teams_df = mens_teams if is_mens else womens_teams
    result = []
    
    for i, (team_id, stats) in enumerate(sweet_16):
        team_name = get_team_name(team_id, teams_df)
        result.append({
            'Rank': i + 1,
            'TeamID': team_id,
            'Team': team_name,
            'Average Win Probability': stats['avg_prob']
        })
    
    return pd.DataFrame(result)

# Main app layout
st.sidebar.title("Options")
display_option = st.sidebar.radio(
    "Display Option",
    ["Generate Full Submission", "View Sweet 16 Teams", "View Sample Predictions"]
)

# Try to load existing submission - only if not already loaded
if not st.session_state.data_loaded:
    existing_predictions_df = load_existing_submission()
    if existing_predictions_df is not None:
        st.session_state.predictions_df = existing_predictions_df
        st.success(f"Loaded existing submission with {len(existing_predictions_df)} predictions")
        st.session_state.data_loaded = True

if display_option == "Generate Full Submission":
    st.header("Generate Submission File")
    
    if st.session_state.predictions_df is not None:
        st.info("A submission file has already been loaded. You can regenerate predictions or use the existing ones.")
    else:
        st.write("This will predict all matchups in SampleSubmissionStage2.csv and create a submission file.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Generate submission
        if st.button("Generate New Submission", use_container_width=True):
            with st.spinner("Generating predictions for all matchups..."):
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Call the prediction function
                predictions_df = predict_all_matchups()
                
                # Store in session state
                st.session_state.predictions_df = predictions_df
                st.session_state.data_loaded = True
                
                # Set progress to completion
                progress_bar.progress(1.0)
                
                st.success(f"Submission file created with {len(predictions_df)} predictions")
                
                # Download button
                st.download_button(
                    label="Download Submission CSV",
                    data=predictions_df[['ID', 'Pred']].to_csv(index=False).encode('utf-8'),
                    file_name="submission.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    with col2:
        st.info("This will predict outcomes for all 131,407 potential matchups between NCAA basketball teams. The process may take a few minutes.")
    
    # Display detailed table with all predictions if they've been generated
    if st.session_state.predictions_df is not None:
        st.subheader("All Predicted Matchups")
        
        # Add a search box
        search_term = st.text_input("Search for a team name", "")
        
        filtered_df = st.session_state.predictions_df
        if search_term:
            filtered_df = st.session_state.predictions_df[
                st.session_state.predictions_df['Team1'].str.contains(search_term, case=False) | 
                st.session_state.predictions_df['Team2'].str.contains(search_term, case=False)
            ]
        
        # Display table with team names and probabilities
        display_cols = ['ID', 'Team1', 'Team2', 'Team1_Win_Probability']
        st.dataframe(filtered_df[display_cols], use_container_width=True)
        
        # Add download button
        st.download_button(
            label="Download Submission CSV",
            data=st.session_state.predictions_df[['ID', 'Pred']].to_csv(index=False).encode('utf-8'),
            file_name="submission.csv",
            mime="text/csv"
        )

elif display_option == "View Sweet 16 Teams":
    st.header("Predicted Sweet 16 Teams")
    
    if st.session_state.predictions_df is None:
        st.warning("Please generate predictions first or load an existing submission file.")
    else:
        # Create tabs for Men's and Women's tournaments
        tab1, tab2 = st.tabs(["Men's Sweet 16", "Women's Sweet 16"])
        
        with tab1:
            st.subheader("Predicted Men's Sweet 16 Teams")
            mens_sweet_16 = get_sweet_16_teams(st.session_state.predictions_df, is_mens=True)
            st.dataframe(mens_sweet_16, use_container_width=True)
            
            # Add a colorful display of the teams
            st.write("### Men's Sweet 16 Teams")
            cols = st.columns(4)
            for i, row in mens_sweet_16.iterrows():
                with cols[i % 4]:
                    st.markdown(f"""
                    <div style="background-color:#13294B; color:white; padding:10px; border-radius:5px; margin:5px; text-align:center;">
                        <h3 style="margin:0; color:white;">{row['Team']}</h3>
                        <p style="margin:0; color:#C8102E;">Rank: {row['Rank']}</p>
                        <p style="margin:0;">Win Prob: {row['Average Win Probability']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Predicted Women's Sweet 16 Teams")
            womens_sweet_16 = get_sweet_16_teams(st.session_state.predictions_df, is_mens=False)
            st.dataframe(womens_sweet_16, use_container_width=True)
            
            # Add a colorful display of the teams
            st.write("### Women's Sweet 16 Teams")
            cols = st.columns(4)
            for i, row in womens_sweet_16.iterrows():
                with cols[i % 4]:
                    st.markdown(f"""
                    <div style="background-color:#13294B; color:white; padding:10px; border-radius:5px; margin:5px; text-align:center;">
                        <h3 style="margin:0; color:white;">{row['Team']}</h3>
                        <p style="margin:0; color:#C8102E;">Rank: {row['Rank']}</p>
                        <p style="margin:0;">Win Prob: {row['Average Win Probability']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
else:  # View Sample Predictions
    st.header("View Sample Predictions")
    
    # Men's teams
    st.subheader("Men's Teams Matchup Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Select Team 1 (Men's)", mens_teams['TeamName'].tolist())
        team1_id = mens_teams[mens_teams['TeamName'] == team1]['TeamID'].values[0]
    
    with col2:
        team2 = st.selectbox("Select Team 2 (Men's)", [t for t in mens_teams['TeamName'].tolist() if t != team1])
        team2_id = mens_teams[mens_teams['TeamName'] == team2]['TeamID'].values[0]
    
    if st.button("Predict Men's Matchup"):
        prob = predict_matchup(team1_id, team2_id, mens_processor, mens_model, mens_pipeline, True)
        
        # Create a DataFrame for this prediction
        matchup_df = pd.DataFrame([{
            'ID': f"2025_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}",
            'Team1': team1,
            'Team2': team2,
            'Team1_Win_Probability': prob if team1_id < team2_id else (1 - prob)
        }])
        
        st.dataframe(matchup_df, use_container_width=True)
    
    # Women's teams
    st.subheader("Women's Teams Matchup Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1_w = st.selectbox("Select Team 1 (Women's)", womens_teams['TeamName'].tolist())
        team1_id_w = womens_teams[womens_teams['TeamName'] == team1_w]['TeamID'].values[0]
    
    with col2:
        team2_w = st.selectbox("Select Team 2 (Women's)", [t for t in womens_teams['TeamName'].tolist() if t != team1_w])
        team2_id_w = womens_teams[womens_teams['TeamName'] == team2_w]['TeamID'].values[0]
    
    if st.button("Predict Women's Matchup"):
        prob = predict_matchup(team1_id_w, team2_id_w, womens_processor, womens_model, womens_pipeline, False)
        
        # Create a DataFrame for this prediction
        matchup_df = pd.DataFrame([{
            'ID': f"2025_{min(team1_id_w, team2_id_w)}_{max(team1_id_w, team2_id_w)}",
            'Team1': team1_w,
            'Team2': team2_w,
            'Team1_Win_Probability': prob if team1_id_w < team2_id_w else (1 - prob)
        }])
        
        st.dataframe(matchup_df, use_container_width=True)

# About section
st.markdown("---")
st.markdown("### About the Model")
st.markdown("""
This app uses LightGBM models trained on historical NCAA basketball data to predict the outcomes 
of matchups in the NCAA Men's and Women's Basketball Tournament 2025. The predictions are based 
on team statistics from the 2025 regular season.

**Key features used in prediction:**
- Team win rates
- Points scored and allowed
- Field goal percentages
- Rebounding statistics
- And many more team performance metrics
""")
