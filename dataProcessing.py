import pandas as pd
import os

class BasketballDataProcessor:
    def __init__(self, data_dir: str, is_mens: bool):
        """
        Initialize the processor with a data directory and gender specification.
        
        Args:
            data_dir (str): Directory containing the CSV files.
            is_mens (bool): True for men's data, False for women's data.
        """
        self.data_dir = data_dir
        self.is_mens = is_mens
        self.prefix = 'M' if is_mens else 'W'
        
        # DataFrames for raw data
        self.teams_df = None
        self.seasons_df = None
        self.regular_season_results = None
        self.regular_season_detailed = None
        self.tourney_results = None
        self.tourney_detailed = None
        self.tourney_seeds = None
        
        # Caches for precomputed statistics
        self.season_stats_cache = {}
        self.detailed_stats_cache = {}
        
    def load_data(self):
        """Load all relevant CSV files into DataFrames with error handling."""
        print(f"Loading {'men''s' if self.is_mens else 'women''s'} basketball data...")
        
        try:
            # Load basic data
            self.teams_df = pd.read_csv(os.path.join(self.data_dir, f'{self.prefix}Teams.csv'))
            self.seasons_df = pd.read_csv(os.path.join(self.data_dir, f'{self.prefix}Seasons.csv'))
            
            # Load game results
            self.regular_season_results = pd.read_csv(
                os.path.join(self.data_dir, f'{self.prefix}RegularSeasonCompactResults.csv'))
            
            try:
                self.regular_season_detailed = pd.read_csv(
                    os.path.join(self.data_dir, f'{self.prefix}RegularSeasonDetailedResults.csv'))
            except FileNotFoundError:
                print(f"Warning: {self.prefix}RegularSeasonDetailedResults.csv not found.")
                self.regular_season_detailed = pd.DataFrame()
            
            # Load tournament data
            self.tourney_results = pd.read_csv(
                os.path.join(self.data_dir, f'{self.prefix}NCAATourneyCompactResults.csv'))
            
            try:
                self.tourney_detailed = pd.read_csv(
                    os.path.join(self.data_dir, f'{self.prefix}NCAATourneyDetailedResults.csv'))
            except FileNotFoundError:
                print(f"Warning: {self.prefix}NCAATourneyDetailedResults.csv not found.")
                self.tourney_detailed = pd.DataFrame()
                
            self.tourney_seeds = pd.read_csv(
                os.path.join(self.data_dir, f'{self.prefix}NCAATourneySeeds.csv'))
            
            print("Data loading completed.")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def precompute_season_stats(self, season: int):
        """
        Precompute basic team statistics for a given season and store in cache.
        
        Args:
            season (int): The season year to process.
        """
        if season in self.season_stats_cache:
            return
        
        # Filter games for the season
        season_games = self.regular_season_results[self.regular_season_results['Season'] == season]
        team_stats = {}
        
        for team_id in self.teams_df['TeamID'].unique():
            team_wins = season_games[season_games['WTeamID'] == team_id]
            team_losses = season_games[season_games['LTeamID'] == team_id]
            total_games = len(team_wins) + len(team_losses)
            
            if total_games == 0:
                continue
                
            # Calculate basic statistics
            wins = len(team_wins)
            win_rate = wins / total_games
            points_scored = (team_wins['WScore'].sum() + team_losses['LScore'].sum()) / total_games
            points_allowed = (team_wins['LScore'].sum() + team_losses['WScore'].sum()) / total_games
            
            team_stats[team_id] = {
                'WinRate': win_rate,
                'TotalGames': total_games,
                'PointsScored': points_scored,
                'PointsAllowed': points_allowed,
                'PointsDiff': points_scored - points_allowed
            }
        
        self.season_stats_cache[season] = pd.DataFrame.from_dict(team_stats, orient='index')

    def precompute_detailed_stats(self, season: int):
        """
        Precompute detailed team statistics for a given season and store in cache.
        
        Args:
            season (int): The season year to process.
        """
        if season in self.detailed_stats_cache or self.regular_season_detailed.empty:
            return
        
        season_games = self.regular_season_detailed[self.regular_season_detailed['Season'] == season]
        team_stats = {}
        
        for team_id in self.teams_df['TeamID'].unique():
            team_wins = season_games[season_games['WTeamID'] == team_id]
            team_losses = season_games[season_games['LTeamID'] == team_id]
            total_games = len(team_wins) + len(team_losses)
            
            if total_games == 0:
                continue
                
            # Aggregate detailed stats for wins
            w_stats = {
                'FGM': team_wins['WFGM'].sum(),
                'FGA': team_wins['WFGA'].sum(),
                'FGM3': team_wins['WFGM3'].sum(),
                'FGA3': team_wins['WFGA3'].sum(),
                'FTM': team_wins['WFTM'].sum(),
                'FTA': team_wins['WFTA'].sum(),
                'OR': team_wins['WOR'].sum(),
                'DR': team_wins['WDR'].sum()
            }
            
            # Aggregate detailed stats for losses
            l_stats = {
                'FGM': team_losses['LFGM'].sum(),
                'FGA': team_losses['LFGA'].sum(),
                'FGM3': team_losses['LFGM3'].sum(),
                'FGA3': team_losses['LFGA3'].sum(),
                'FTM': team_losses['LFTM'].sum(),
                'FTA': team_losses['LFTA'].sum(),
                'OR': team_losses['LOR'].sum(),
                'DR': team_losses['LDR'].sum()
            }
            
            # Combine and average stats
            combined_stats = {}
            for stat in w_stats.keys():
                total = w_stats[stat] + l_stats[stat]
                combined_stats[f'{stat}_pg'] = total / total_games
                
            # Calculate shooting percentages
            combined_stats['FG_pct'] = (w_stats['FGM'] + l_stats['FGM']) / (w_stats['FGA'] + l_stats['FGA']) if (w_stats['FGA'] + l_stats['FGA']) > 0 else 0
            combined_stats['FG3_pct'] = (w_stats['FGM3'] + l_stats['FGM3']) / (w_stats['FGA3'] + l_stats['FGA3']) if (w_stats['FGA3'] + l_stats['FGA3']) > 0 else 0
            combined_stats['FT_pct'] = (w_stats['FTM'] + l_stats['FTM']) / (w_stats['FTA'] + l_stats['FTA']) if (w_stats['FTA'] + l_stats['FTA']) > 0 else 0
            
            team_stats[team_id] = combined_stats
        
        self.detailed_stats_cache[season] = pd.DataFrame.from_dict(team_stats, orient='index')

    def create_matchup_features(self, team1_id: int, team2_id: int, season: int) -> dict:
        """
        Create features for a matchup using precomputed statistics.
        
        Args:
            team1_id (int): ID of the first team.
            team2_id (int): ID of the second team.
            season (int): The season year.
            
        Returns:
            dict: Features for the matchup or None if stats are unavailable.
        """
        season_stats = self.season_stats_cache.get(season)
        if season_stats is None or team1_id not in season_stats.index or team2_id not in season_stats.index:
            return None
        
        team1_stats = season_stats.loc[team1_id]
        team2_stats = season_stats.loc[team2_id]
        
        # Basic matchup features
        features = {
            'Season': season,
            'Team1ID': team1_id,
            'Team2ID': team2_id,
            'Team1WinRate': team1_stats['WinRate'],
            'Team2WinRate': team2_stats['WinRate'],
            'Team1PointsScored': team1_stats['PointsScored'],
            'Team2PointsScored': team2_stats['PointsScored'],
            'Team1PointsAllowed': team1_stats['PointsAllowed'],
            'Team2PointsAllowed': team2_stats['PointsAllowed'],
            'Team1PointsDiff': team1_stats['PointsDiff'],
            'Team2PointsDiff': team2_stats['PointsDiff'],
            'WinRateDiff': team1_stats['WinRate'] - team2_stats['WinRate'],
            'PointsScoredDiff': team1_stats['PointsScored'] - team2_stats['PointsScored']
        }
        
        # Add detailed stats if available
        detailed_stats = self.detailed_stats_cache.get(season)
        if detailed_stats is not None and team1_id in detailed_stats.index and team2_id in detailed_stats.index:
            team1_detailed = detailed_stats.loc[team1_id]
            team2_detailed = detailed_stats.loc[team2_id]
            features.update({f"Team1_{col}": team1_detailed[col] for col in team1_detailed.index})
            features.update({f"Team2_{col}": team2_detailed[col] for col in team2_detailed.index})
        
        return features

    def create_training_dataset(self, start_season: int, end_season: int) -> pd.DataFrame:
        """
        Create a training dataset from historical games using precomputed stats.
        
        Args:
            start_season (int): Starting season year.
            end_season (int): Ending season year.
            
        Returns:
            pd.DataFrame: Dataset with matchup features and target variable.
        """
        all_matchups = []
        
        for season in range(start_season, end_season + 1):
            # Precompute statistics for the season
            self.precompute_season_stats(season)
            self.precompute_detailed_stats(season)
            
            season_games = self.regular_season_results[self.regular_season_results['Season'] == season]
            
            for _, game in season_games.iterrows():
                # Team1 as winner, Team2 as loser
                features = self.create_matchup_features(game['WTeamID'], game['LTeamID'], season)
                if features is not None:
                    features['Target'] = 1  # Team1 won
                    all_matchups.append(features)
                
                # Team1 as loser, Team2 as winner
                features = self.create_matchup_features(game['LTeamID'], game['WTeamID'], season)
                if features is not None:
                    features['Target'] = 0  # Team2 won
                    all_matchups.append(features)
        
        return pd.DataFrame(all_matchups)

def main():
    """Main function to process men's and women's basketball data."""
    data_dir = "reducedDataset"  # Specify your data directory here
    
    # Process men's data
    mens_processor = BasketballDataProcessor(data_dir, is_mens=True)
    mens_processor.load_data()
    mens_dataset = mens_processor.create_training_dataset(2010, 2024)
    mens_dataset.to_csv('mens_training_data.csv', index=False)
    print("Men's training data saved to 'mens_training_data.csv'.")
    
    # Process women's data
    womens_processor = BasketballDataProcessor(data_dir, is_mens=False)
    womens_processor.load_data()
    womens_dataset = womens_processor.create_training_dataset(2010, 2024)
    womens_dataset.to_csv('womens_training_data.csv', index=False)
    print("Women's training data saved to 'womens_training_data.csv'.")

if __name__ == "__main__":
    main()