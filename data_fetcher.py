import requests
from datetime import datetime, timedelta
import pandas as pd
import time
import numpy as np

class FootballDataAPI:
    BASE_URL = "https://api.football-data.org/v4"
    
    def __init__(self, api_key):
        self.headers = {"X-Auth-Token": api_key}
    
    def get_pl_matches(self, season=None):
        """Fetch Premier League matches for a given season"""
        url = f"{self.BASE_URL}/competitions/PL/matches"
        if season:
            url += f"?season={season}"
        
        response = requests.get(url, headers=self.headers)
        # Add rate limiting to avoid API restrictions
        time.sleep(1)
        return response.json()
    
    def get_team_stats(self, team_id, season=None):
        """Fetch team statistics"""
        url = f"{self.BASE_URL}/teams/{team_id}"
        params = {"season": season} if season else {}
        response = requests.get(url, headers=self.headers, params=params)
        time.sleep(1)
        return response.json()
    
    def get_upcoming_matches(self):
        """Fetch upcoming Premier League matches"""
        today = datetime.now().strftime("%Y-%m-%d")
        next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        
        url = f"{self.BASE_URL}/competitions/PL/matches"
        params = {
            "dateFrom": today,
            "dateTo": next_week,
            "status": "SCHEDULED"
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        time.sleep(1)  # Rate limiting
        return response.json()

    def get_team_recent_stats(self, team_id):
        """Fetch recent team statistics"""
        url = f"{self.BASE_URL}/teams/{team_id}/matches"
        params = {
            "limit": 5,  # Last 5 matches
            "status": "FINISHED"
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        time.sleep(1)
        return response.json()

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

def prepare_match_data(api):
    """Prepare historical match data with enhanced features"""
    current_year = datetime.now().year
    matches = []
    
    # Fetch last 8 seasons
    for season in range(current_year - 7, current_year + 1):
        print(f"Fetching season {season}")
        season_matches = api.get_pl_matches(season)
        if 'matches' in season_matches:
            matches.extend(season_matches['matches'])
        time.sleep(1)
    
    match_data = []
    team_form = {}
    team_stats = {}
    team_positions = {}
    season_stats = {}
    
    # Pre-process all team names
    all_teams = set()
    for match in matches:
        home_team = clean_team_name(match['homeTeam']['name'])
        away_team = clean_team_name(match['awayTeam']['name'])
        all_teams.add(home_team)
        all_teams.add(away_team)
    
    # Sort matches by date for proper season tracking
    sorted_matches = sorted(matches, key=lambda x: x['utcDate'])
    current_season = None
    
    for match in sorted_matches:
        if match['status'] == 'FINISHED':
            season_id = match['season']['id']
            match_date = datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ")
            
            # Initialize new season stats if needed
            if season_id != current_season:
                current_season = season_id
                # Initialize stats for all teams with cleaned names
                season_stats[season_id] = {team: {
                    'points': 0,
                    'goals_for': 0,
                    'goals_against': 0,
                    'wins': 0,
                    'draws': 0,
                    'losses': 0,
                    'red_cards': 0,
                    'clean_sheets': 0,
                    'matches_played': 0
                } for team in all_teams}  # Use pre-processed team names
                
                # Reset team positions for new season
                team_positions[season_id] = {}
            
            # Clean team names
            home_team = clean_team_name(match['homeTeam']['name'])
            away_team = clean_team_name(match['awayTeam']['name'])
            
            # Get match statistics
            home_score = match['score']['fullTime']['home']
            away_score = match['score']['fullTime']['away']
            home_half_score = match['score'].get('halfTime', {}).get('home', 0)
            away_half_score = match['score'].get('halfTime', {}).get('away', 0)
            
            # Get detailed match statistics
            home_stats = match.get('homeTeam', {}).get('statistics', {})
            away_stats = match.get('awayTeam', {}).get('statistics', {})
            
            # Update season statistics
            for team, is_home in [(home_team, True), (away_team, False)]:
                score = home_score if is_home else away_score
                conceded = away_score if is_home else home_score
                season_stats[season_id][team]['matches_played'] += 1
                season_stats[season_id][team]['goals_for'] += score
                season_stats[season_id][team]['goals_against'] += conceded
                
                if score > conceded:
                    season_stats[season_id][team]['points'] += 3
                    season_stats[season_id][team]['wins'] += 1
                elif score == conceded:
                    season_stats[season_id][team]['points'] += 1
                    season_stats[season_id][team]['draws'] += 1
                else:
                    season_stats[season_id][team]['losses'] += 1
                
                if conceded == 0:
                    season_stats[season_id][team]['clean_sheets'] += 1
            
            # Calculate current league positions
            season_table = sorted(
                season_stats[season_id].items(),
                key=lambda x: (-x[1]['points'], -(x[1]['goals_for'] - x[1]['goals_against']), -x[1]['goals_for'])
            )
            team_positions[season_id] = {team: pos+1 for pos, (team, _) in enumerate(season_table)}
            
            # Prepare match data with enhanced features
            match_data.append({
                # Basic match information
                'season': season_id,
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                
                # Full time results
                'home_score': home_score,
                'away_score': away_score,
                'result': 1 if home_score > away_score else 0 if home_score == away_score else -1,
                
                # Half time results
                'ht_home_score': home_half_score,
                'ht_away_score': away_half_score,
                'ht_result': 1 if home_half_score > away_half_score else 0 if home_half_score == away_half_score else -1,
                
                # Team positions
                'home_position': team_positions[season_id][home_team],
                'away_position': team_positions[season_id][away_team],
                'position_difference': team_positions[season_id][home_team] - team_positions[season_id][away_team],
                
                # Season performance metrics
                'home_points': season_stats[season_id][home_team]['points'],
                'away_points': season_stats[season_id][away_team]['points'],
                'home_goal_diff': season_stats[season_id][home_team]['goals_for'] - season_stats[season_id][home_team]['goals_against'],
                'away_goal_diff': season_stats[season_id][away_team]['goals_for'] - season_stats[season_id][away_team]['goals_against'],
                
                # Form metrics (last 5 games)
                'home_form': safe_rate(sum(team_form.get(home_team, [])[-5:]), min(len(team_form.get(home_team, [])), 5), 0.5),
                'away_form': safe_rate(sum(team_form.get(away_team, [])[-5:]), min(len(team_form.get(away_team, [])), 5), 0.5),
                
                # Performance metrics
                'home_win_rate': safe_rate(season_stats[season_id][home_team]['wins'], 
                                         season_stats[season_id][home_team]['matches_played'], 0.5),
                'away_win_rate': safe_rate(season_stats[season_id][away_team]['wins'],
                                         season_stats[season_id][away_team]['matches_played'], 0.5),
                
                # Defensive metrics
                'home_clean_sheet_rate': safe_rate(season_stats[season_id][home_team]['clean_sheets'],
                                                 season_stats[season_id][home_team]['matches_played'], 0.3),
                'away_clean_sheet_rate': safe_rate(season_stats[season_id][away_team]['clean_sheets'],
                                                 season_stats[season_id][away_team]['matches_played'], 0.3),
                
                # Goals metrics
                'home_avg_goals_scored': safe_rate(season_stats[season_id][home_team]['goals_for'],
                                                 season_stats[season_id][home_team]['matches_played'], 1.5),
                'away_avg_goals_scored': safe_rate(season_stats[season_id][away_team]['goals_for'],
                                                 season_stats[season_id][away_team]['matches_played'], 1.5),
                'home_avg_goals_conceded': safe_rate(season_stats[season_id][home_team]['goals_against'],
                                                   season_stats[season_id][home_team]['matches_played'], 1.5),
                'away_avg_goals_conceded': safe_rate(season_stats[season_id][away_team]['goals_against'],
                                                   season_stats[season_id][away_team]['matches_played'], 1.5)
            })
            
            # Update form
            home_form_value = 1 if home_score > away_score else 0.5 if home_score == away_score else 0
            away_form_value = 1 if away_score > home_score else 0.5 if home_score == away_score else 0
            team_form.setdefault(home_team, []).append(home_form_value)
            team_form.setdefault(away_team, []).append(away_form_value)
    
    # Convert to DataFrame and handle missing values
    df = pd.DataFrame(match_data)
    df = handle_missing_values(df)
    
    return df

def handle_missing_values(df):
    """Handle missing values in the DataFrame"""
    # Identify numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Replace infinities and fill NaNs only for numeric columns
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Fill NaNs in non-numeric columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_columns:
        if col == 'date':
            continue  # Skip date column
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def safe_rate(numerator, denominator, default=0.0):
    """Safely calculate a rate"""
    if denominator == 0 or denominator is None:
        return default
    rate = numerator / denominator
    # Clip rate between 0 and 1
    return np.clip(rate, 0, 1)