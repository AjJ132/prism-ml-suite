




from logging import Logger

from sklearn.discriminant_analysis import StandardScaler
from database.db_manager import get_db
from database.models.vw_player_passing_stats import VwPlayerPassingStats
from database.models.vw_player_receiving_stats import VwPlayerReceivingStats
from database.models.vw_player_rushing_stats import VwPlayerRushingStats
from database.models.vw_players import VwPlayers
import pandas as pd

from util.helpers.get_team_division import load_team_divisions

class DataEngineeringPipeline:
    def __init__(self, logger: Logger, mode: int, upcoming_season,  raw_data_path: str = "data/raw", engineered_data_path: str = "data/engineered"):
        self.logger = logger
        self.mode = mode
        self.upcoming_season = upcoming_season
        self.raw_data_path = raw_data_path
        self.engineered_data_path = engineered_data_path
        self.players = None
        self.teams = load_team_divisions()
        self.all_players = pd.read_csv(raw_data_path + "/all_college_players.csv")
        
    def generate_future_season_entries(self, stats_df, position_type):
        """
        Generate entries for the upcoming season based on players' eligibility and previous stats
        """
        # Get all unique players from the previous season
        previous_season = stats_df['season'].max()
        previous_season_players = stats_df[stats_df['season'] == previous_season]['master_player_id'].unique()
        
        # Filter players who are still eligible (need to implement eligibility logic)
        eligible_players = []
        for player_id in previous_season_players:
            player_seasons = self.all_players[self.all_players['master_player_id'] == player_id]['inferred_season'].unique()
            if len(player_seasons) < 4:  # Basic eligibility check (can be made more sophisticated)
                eligible_players.append(player_id)
        
        # Create future season entries
        future_entries = []
        for player_id in eligible_players:
            # Create a base entry with player ID and future season
            entry = {
                'master_player_id': player_id,
                'season': self.upcoming_season
            }
            
            # Add previous season stats as a starting point
            player_stats = stats_df[stats_df['master_player_id'] == player_id].sort_values('season').iloc[-1]
            
            # Copy over relevant columns based on position type
            if position_type == 'qb_passing':
                entry['passing_yards'] = None  # Target to predict
                for col in ['completions', 'attempts', 'interceptions', 'completion_percentage', 
                           'yards_per_attempt', 'passing_tds']:
                    if f'previous_{col}' in player_stats:
                        entry[f'previous_{col}'] = player_stats[col]
            
            elif position_type == 'qb_rushing':
                entry['rush_yards'] = None  # Target to predict
                for col in ['rush_attempts', 'rush_tds', 'yards_per_carry']:
                    if f'previous_{col}' in player_stats:
                        entry[f'previous_{col}'] = player_stats[col]
            
            elif position_type == 'rb_rushing':
                entry['rush_yards'] = None  # Target to predict
                for col in ['rush_attempts', 'rush_tds', 'yards_per_carry']:
                    if f'previous_{col}' in player_stats:
                        entry[f'previous_{col}'] = player_stats[col]
            
            elif position_type == 'rb_receiving' or position_type == 'wr_receiving':
                entry['receiving_yards'] = None  # Target to predict
                for col in ['receptions', 'receiving_tds', 'yards_per_reception']:
                    if f'previous_{col}' in player_stats:
                        entry[f'previous_{col}'] = player_stats[col]
            
            # Add EWMA values from previous season
            for col in player_stats.index:
                if '_ewma_' in col:
                    entry[col] = player_stats[col]
            
            # Add division
            entry['division'] = self.get_player_division(player_id, previous_season)
            
            future_entries.append(entry)
        
        # Convert to DataFrame and combine with original stats
        future_df = pd.DataFrame(future_entries)
        return pd.concat([stats_df, future_df], ignore_index=True)
        
    def add_ewma_features(self, df, columns, target_column):
        """
        Add EWMA features for specified columns, including the target column
        """
        result_df = df.copy()
        
        for col in columns:
            result_df[f'{col}_ewma_2'] = (df.groupby('master_player_id')[col]
                                        .transform(lambda x: x.shift(1).ewm(span=2, min_periods=1).mean()))
    
        return result_df
        
    def normalize_features(self, df, target_column, exclude_columns=['master_player_id', 'season']):
        """
        Normalize features using StandardScaler, excluding the target column
        """
        scaler = StandardScaler()
        # Add target column to exclude list
        all_exclude_columns = exclude_columns + [target_column]
        columns_to_normalize = [col for col in df.columns if col not in all_exclude_columns]
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        return df
        
    def get_player_division(self, player_id, season) -> float:
        #get players team id based on season and player_id from all_players
        team_id_values = self.all_players[(self.all_players['master_player_id'] == player_id) & (self.all_players['inferred_season'] == season)]['ap_team_id'].values
        
        if len(team_id_values) == 0:
            return 0   
        
        team_id = team_id_values[0]
        
        #if team_id is not found, return 0 (FCS)
        if team_id is None:
            return 0
        
        #get division based on team_id
        division_values = self.teams[self.teams['espn_team_id'] == team_id]['division'].values
        
        if len(division_values) == 0:
            return 0
        
        division = division_values[0]
        
        if division is None:
            return 0
        
        if division == 'FCS':
            return 0
        elif division == 'G5':
            return 0.5
        elif division == 'P5':
            return 1
        
    def engineer_qbs(self):
        self.logger.info("Engineering QB Stats")
        
        # Load passing and rushing stats
        raw_passing_stats = pd.read_csv(self.raw_data_path + "/stats/qb_passing_stats.csv")
        raw_rushing_stats = pd.read_csv(self.raw_data_path + "/stats/qb_rushing_stats.csv")
        
        # Generate future season entries
        raw_passing_stats = self.generate_future_season_entries(raw_passing_stats, 'qb_passing')
        raw_rushing_stats = self.generate_future_season_entries(raw_rushing_stats, 'qb_rushing')
        
        # Define passing columns
        passing_columns = [
            'completions', 'attempts', 'interceptions', 
            'completion_percentage', 'yards_per_attempt', 'passing_tds',
            'passing_yards'
        ]
        
        # Define rushing columns
        rushing_columns = [
            'rush_attempts', 'rush_tds', 'yards_per_carry',
            'rush_yards'
        ]
        
        # Create lists of previous season columns
        previous_passing_columns = [f'previous_{col}' for col in passing_columns]
        previous_rushing_columns = [f'previous_{col}' for col in rushing_columns]
        
        # Keep base columns including target variables
        passing_base_columns = ['master_player_id', 'season', 'passing_yards']
        rushing_base_columns = ['master_player_id', 'season', 'rush_yards']
        
        # Create DataFrame with base columns
        filtered_passing_stats = raw_passing_stats[passing_base_columns].copy()
        filtered_rushing_stats = raw_rushing_stats[rushing_base_columns].copy()
        
        # Add previous season stats for passing metrics
        for curr_col, prev_col in zip(passing_columns, previous_passing_columns):
            filtered_passing_stats[prev_col] = raw_passing_stats.groupby('master_player_id')[curr_col].shift(1)
            
        # Add previous season stats for rushing metrics
        for curr_col, prev_col in zip(rushing_columns, previous_rushing_columns):
            filtered_rushing_stats[prev_col] = raw_rushing_stats.groupby('master_player_id')[curr_col].shift(1)
        
        # Add EWMA features for both passing and rushing
        passing_ewma_columns = passing_columns
        ewma_stats_passing = self.add_ewma_features(raw_passing_stats[['master_player_id', 'season'] + passing_ewma_columns], passing_ewma_columns,'passing_yards')
        
        rushing_ewma_columns = rushing_columns
        ewma_stats_rushing = self.add_ewma_features(raw_rushing_stats[['master_player_id', 'season'] + rushing_ewma_columns], rushing_ewma_columns, 'rush_yards')
        
        
        # Merge EWMA features back with original filtered stats
        filtered_passing_stats = pd.merge(
            filtered_passing_stats, 
            ewma_stats_passing.drop(passing_ewma_columns, axis=1), 
            on=['master_player_id', 'season'],
            how='left'
        )
        
        filtered_rushing_stats = pd.merge(
            filtered_rushing_stats, 
            ewma_stats_rushing.drop(rushing_ewma_columns, axis=1), 
            on=['master_player_id', 'season'],
            how='left'
        )
        
        # Fill NaN values with 0 for previous season stats and EWMA
        for col in filtered_passing_stats.columns:
            if col not in ['master_player_id', 'season']:
                filtered_passing_stats[col] = filtered_passing_stats[col].fillna(0)
                
        for col in filtered_rushing_stats.columns:
            if col not in ['master_player_id', 'season']:
                filtered_rushing_stats[col] = filtered_rushing_stats[col].fillna(0)
                
        #where target feature is less than 0, set to 0
        filtered_passing_stats[filtered_passing_stats['passing_yards'] < 0] = 0
        filtered_rushing_stats[filtered_rushing_stats['rush_yards'] < 0] = 0
        
        
        # Add player divisions
        filtered_passing_stats['division'] = filtered_passing_stats.apply(
            lambda x: self.get_player_division(x['master_player_id'], x['season']), 
            axis=1
        )
        
        filtered_rushing_stats['division'] = filtered_rushing_stats.apply(
            lambda x: self.get_player_division(x['master_player_id'], x['season']),
            axis=1
        )
        
        # Normalize features
        normalized_passing_stats = self.normalize_features(filtered_passing_stats, target_column='passing_yards')
        normalized_rushing_stats = self.normalize_features(filtered_rushing_stats, target_column='rush_yards')
        
        # Group by master player id and season
        normalized_passing_stats = normalized_passing_stats.groupby(['master_player_id', 'season']).agg({
            'passing_yards': 'sum',
            **{col: 'first' for col in normalized_passing_stats.columns if col not in ['master_player_id', 'season', 'passing_yards']}
        }).reset_index()
        
        normalized_rushing_stats = normalized_rushing_stats.groupby(['master_player_id', 'season']).agg({
            'rush_yards': 'sum',
            **{col: 'first' for col in normalized_rushing_stats.columns if col not in ['master_player_id', 'season', 'rush_yards']}
        }).reset_index()
        
        # Save to csv
        normalized_passing_stats.to_csv(self.engineered_data_path + "/qb_passing_stats.csv", index=False)
        normalized_rushing_stats.to_csv(self.engineered_data_path + "/qb_rushing_stats.csv", index=False)
        
    def engineer_rbs(self):
        self.logger.info("Engineering RB Stats")
    
        # Load rushing and receiving stats
        raw_rushing_stats = pd.read_csv(self.raw_data_path + "/stats/rb_rushing_stats.csv")
        raw_receiving_stats = pd.read_csv(self.raw_data_path + "/stats/rb_receiving_stats.csv")
        
        # Generate future season entries
        raw_rushing_stats = self.generate_future_season_entries(raw_rushing_stats, 'rb_rushing')
        raw_receiving_stats = self.generate_future_season_entries(raw_receiving_stats, 'rb_receiving')
        
        # Define rushing columns
        rushing_columns = [
            'rush_attempts', 'rush_tds', 'yards_per_carry',
            'rush_yards'
        ]
        
        # Define receiving columns
        receiving_columns = [
            'receptions', 'receiving_tds', 'yards_per_reception',
            'receiving_yards'
        ]
        
        # Create lists of previous season columns
        previous_rushing_columns = [f'previous_{col}' for col in rushing_columns]
        previous_receiving_columns = [f'previous_{col}' for col in receiving_columns]
        
        # Keep base columns including target variable
        rushing_base_columns = ['master_player_id', 'season', 'rush_yards']
        receiving_base_columns = ['master_player_id', 'season', 'receiving_yards']
        
        # Create DataFrame with base columns
        filtered_rushing_stats = raw_rushing_stats[rushing_base_columns].copy()
        filtered_receiving_stats = raw_receiving_stats[receiving_base_columns].copy()
        
        # Add previous season stats for rushing metrics
        for curr_col, prev_col in zip(rushing_columns, previous_rushing_columns):
            filtered_rushing_stats[prev_col] = raw_rushing_stats.groupby('master_player_id')[curr_col].shift(1)
        
        # Add previous season stats for receiving metrics
        for curr_col, prev_col in zip(receiving_columns, previous_receiving_columns):
            filtered_receiving_stats[prev_col] = raw_receiving_stats.groupby('master_player_id')[curr_col].shift(1)
        
        # Add EWMA features for both rushing and receiving
        rushing_ewma_columns = rushing_columns
        ewma_stats_rushing = self.add_ewma_features(raw_rushing_stats[['master_player_id', 'season'] + rushing_ewma_columns], rushing_ewma_columns, 'rush_yards')
        
        receiving_ewma_columns = receiving_columns
        ewma_stats_receiving = self.add_ewma_features(raw_receiving_stats[['master_player_id', 'season'] + receiving_ewma_columns], receiving_ewma_columns, 'receiving_yards')
        
        # Merge EWMA features back with original filtered stats
        filtered_rushing_stats = pd.merge(
            filtered_rushing_stats, 
            ewma_stats_rushing.drop(rushing_ewma_columns, axis=1), 
            on=['master_player_id', 'season'],
            how='left'
        )
        
        filtered_receiving_stats = pd.merge(
            filtered_receiving_stats, 
            ewma_stats_receiving.drop(receiving_ewma_columns, axis=1), 
            on=['master_player_id', 'season'],
            how='left'
        )
        
        # Fill NaN values with 0 for previous season stats and EWMA
        for col in filtered_rushing_stats.columns:
            if col not in ['master_player_id', 'season']:
                filtered_rushing_stats[col] = filtered_rushing_stats[col].fillna(0)
                
        for col in filtered_receiving_stats.columns:
            if col not in ['master_player_id', 'season']:
                filtered_receiving_stats[col] = filtered_receiving_stats[col].fillna(0)
                
                
        #where target feature is less than 0, set to 0
        filtered_rushing_stats[filtered_rushing_stats['rush_yards'] < 0] = 0
        filtered_receiving_stats[filtered_receiving_stats['receiving_yards'] < 0] = 0
            

        
        
        # Add player divisions
        filtered_rushing_stats['division'] = filtered_rushing_stats.apply(
            lambda x: self.get_player_division(x['master_player_id'], x['season']), 
            axis=1
        )
        
        filtered_receiving_stats['division'] = filtered_receiving_stats.apply(
            lambda x: self.get_player_division(x['master_player_id'], x['season']),
            axis=1
        )
        
        # Normalize features
        normalized_rushing_stats = self.normalize_features(filtered_rushing_stats, target_column='rush_yards')
        normalized_receiving_stats = self.normalize_features(filtered_receiving_stats, target_column='receiving_yards')
        
        # Group by master player id and season
        normalized_rushing_stats = normalized_rushing_stats.groupby(['master_player_id', 'season']).agg({
            'rush_yards': 'sum',
            **{col: 'first' for col in normalized_rushing_stats.columns if col not in ['master_player_id', 'season', 'rush_yards']}
        }).reset_index()
        
        normalized_receiving_stats = normalized_receiving_stats.groupby(['master_player_id', 'season']).agg({
            'receiving_yards': 'sum',
            **{col: 'first' for col in normalized_receiving_stats.columns if col not in ['master_player_id', 'season', 'receiving_yards']}
        }).reset_index()
        
        # Save to csv
        normalized_rushing_stats.to_csv(self.engineered_data_path + "/rb_rushing_stats.csv", index=False)
        normalized_receiving_stats.to_csv(self.engineered_data_path + "/rb_receiving_stats.csv", index=False)
        
    def engineer_wrs(self):
        self.logger.info("Engineering WR Stats")
        
        # Load receiving stats
        raw_receiving_stats = pd.read_csv(self.raw_data_path + "/stats/wr_receiving_stats.csv")
        
        # Generate future season entries
        raw_receiving_stats = self.generate_future_season_entries(raw_receiving_stats, 'wr_receiving')
        
        # Define columns to keep and create previous season columns
        current_columns = [
            'receptions', 'receiving_tds', 'yards_per_reception',
            'receiving_yards'  # Added receiving_yards to track previous season
        ]
        
        # Create list of previous season columns
        previous_columns = [f'previous_{col}' for col in current_columns]
        
        # Keep base columns including target variable
        base_columns = ['master_player_id', 'season', 'receiving_yards']
        
        # Create DataFrame with base columns
        filtered_receiving_stats = raw_receiving_stats[base_columns].copy()
        
        # Add previous season stats for each metric
        for curr_col, prev_col in zip(current_columns, previous_columns):
            filtered_receiving_stats[prev_col] = raw_receiving_stats.groupby('master_player_id')[curr_col].shift(1)
            
        # Add EWMA features
        ewma_columns = current_columns
        ewma_stats = self.add_ewma_features(raw_receiving_stats[['master_player_id', 'season'] + ewma_columns], 
                                          ewma_columns,
                                          'receiving_yards')
        
        # Merge EWMA features back with original filtered stats
        filtered_receiving_stats = pd.merge(filtered_receiving_stats, 
                                        ewma_stats.drop(ewma_columns, axis=1), 
                                        on=['master_player_id', 'season'],
                                        how='left')
            
        # Fill NaN values with 0 for previous season stats and EWMA
        for col in filtered_receiving_stats.columns:
            if col not in ['master_player_id', 'season']:
                filtered_receiving_stats[col] = filtered_receiving_stats[col].fillna(0)
                
                
        #where target feature is less than 0, set to 0
        filtered_receiving_stats[filtered_receiving_stats['receiving_yards'] < 0] = 0
                
                
        #add player divisions
        filtered_receiving_stats['division'] = filtered_receiving_stats.apply(lambda x: self.get_player_division(x['master_player_id'], x['season']), axis=1)
        
        # Normalize features
        normalized_receiving_stats = self.normalize_features(filtered_receiving_stats, target_column='receiving_yards')
            
        # Group by master player id and season
        normalized_receiving_stats = normalized_receiving_stats.groupby(['master_player_id', 'season']).agg({
            'receiving_yards': 'sum',
            **{col: 'first' for col in normalized_receiving_stats.columns if col not in ['master_player_id', 'season', 'receiving_yards']}
        }).reset_index()
        
        # Save to csv
        normalized_receiving_stats.to_csv(self.engineered_data_path + "/wr_receiving_stats.csv", index=False)
    
    def run(self):
        self.logger.info("Running Data Engineering Pipeline")
        
        self.players = pd.read_csv(self.raw_data_path + "/all_college_players.csv")
        
        # Engineer QB stats
        self.engineer_qbs()
        
        # Engineer RB stats
        self.engineer_rbs()
        
        # Engineer WR stats
        self.engineer_wrs()