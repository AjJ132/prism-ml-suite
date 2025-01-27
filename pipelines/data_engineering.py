




from logging import Logger
from database.db_manager import get_db
from database.models.vw_player_passing_stats import VwPlayerPassingStats
from database.models.vw_player_receiving_stats import VwPlayerReceivingStats
from database.models.vw_player_rushing_stats import VwPlayerRushingStats
from database.models.vw_players import VwPlayers
import pandas as pd

class DataEngineeringPipeline:
    def __init__(self, logger: Logger, mode: int, raw_data_path: str = "data/raw", engineered_data_path: str = "data/engineered"):
        self.logger = logger
        self.mode = mode
        self.raw_data_path = raw_data_path
        self.engineered_data_path = engineered_data_path
        self.players = None
        
    def add_ewma_features(self, df, columns, target_column):
        """
        Add EWMA features for specified columns, including the target column
        """
        result_df = df.copy()
        
        for col in columns:
            result_df[f'{col}_ewma_2'] = (df.groupby('master_player_id')[col]
                                        .transform(lambda x: x.shift(1).ewm(span=2, min_periods=1).mean()))
    
        return result_df
        
    def engineer_qbs(self):
        self.logger.info("Engineering QB Stats")
        
        # Load passing stats
        raw_passing_stats = pd.read_csv(self.raw_data_path + "/stats/qb_passing_stats.csv")
        
        # Define columns to keep and create previous season columns
        current_columns = [
            'completions', 'attempts', 'interceptions', 
            'completion_percentage', 'yards_per_attempt', 'passing_tds',
            'passing_yards'  # Added passing_yards to track previous season
        ]
        
        # Create list of previous season columns
        previous_columns = [f'previous_{col}' for col in current_columns]
        
        # Keep base columns including target variable
        base_columns = ['master_player_id', 'season', 'passing_yards']
        
        # Create DataFrame with base columns
        filtered_passing_stats = raw_passing_stats[base_columns].copy()
        
        # Add previous season stats for each metric
        for curr_col, prev_col in zip(current_columns, previous_columns):
            filtered_passing_stats[prev_col] = raw_passing_stats.groupby('master_player_id')[curr_col].shift(1)
            
        # Add EWMA features
        ewma_columns = current_columns
        ewma_stats = self.add_ewma_features(raw_passing_stats[['master_player_id', 'season'] + ewma_columns], 
                                          ewma_columns,
                                          'passing_yards')
        
        # Merge EWMA features back with original filtered stats
        filtered_passing_stats = pd.merge(filtered_passing_stats, 
                                        ewma_stats.drop(ewma_columns, axis=1), 
                                        on=['master_player_id', 'season'],
                                        how='left')
            
        # Fill NaN values with 0 for previous season stats and EWMA
        for col in filtered_passing_stats.columns:
            if col not in ['master_player_id', 'season']:
                filtered_passing_stats[col] = filtered_passing_stats[col].fillna(0)
            
        # Group by master player id and season
        filtered_passing_stats = filtered_passing_stats.groupby(['master_player_id', 'season']).agg({
            'passing_yards': 'sum',
            **{col: 'first' for col in filtered_passing_stats.columns if col not in ['master_player_id', 'season', 'passing_yards']}
        }).reset_index()
        
        # Save to csv
        filtered_passing_stats.to_csv(self.engineered_data_path + "/qb_passing_stats.csv", index=False)
        
    def engineer_rbs(self):
        self.logger.info("Engineering RB Stats")
        
        # Load rushing stats
        raw_rushing_stats = pd.read_csv(self.raw_data_path + "/stats/rb_rushing_stats.csv")
        
        # Define columns to keep and create previous season columns
        current_columns = [
            'rush_attempts', 'rush_tds', 'yards_per_carry',
            'rush_yards'  # Added rush_yards to track previous season
        ]
        
        # Create list of previous season columns
        previous_columns = [f'previous_{col}' for col in current_columns]
        
        # Keep base columns including target variable
        base_columns = ['master_player_id', 'season', 'rush_yards']
        
        # Create DataFrame with base columns
        filtered_rushing_stats = raw_rushing_stats[base_columns].copy()
        
        # Add previous season stats for each metric
        for curr_col, prev_col in zip(current_columns, previous_columns):
            filtered_rushing_stats[prev_col] = raw_rushing_stats.groupby('master_player_id')[curr_col].shift(1)
            
        # Add EWMA features
        ewma_columns = current_columns
        ewma_stats = self.add_ewma_features(raw_rushing_stats[['master_player_id', 'season'] + ewma_columns], 
                                          ewma_columns,
                                          'rush_yards')
        
        # Merge EWMA features back with original filtered stats
        filtered_rushing_stats = pd.merge(filtered_rushing_stats, 
                                        ewma_stats.drop(ewma_columns, axis=1), 
                                        on=['master_player_id', 'season'],
                                        how='left')
            
        # Fill NaN values with 0 for previous season stats and EWMA
        for col in filtered_rushing_stats.columns:
            if col not in ['master_player_id', 'season']:
                filtered_rushing_stats[col] = filtered_rushing_stats[col].fillna(0)
            
        # Group by master player id and season
        filtered_rushing_stats = filtered_rushing_stats.groupby(['master_player_id', 'season']).agg({
            'rush_yards': 'sum',
            **{col: 'first' for col in filtered_rushing_stats.columns if col not in ['master_player_id', 'season', 'rush_yards']}
        }).reset_index()
        
        # Save to csv
        filtered_rushing_stats.to_csv(self.engineered_data_path + "/rb_rushing_stats.csv", index=False)
        
    def engineer_wrs(self):
        self.logger.info("Engineering WR Stats")
        
        # Load receiving stats
        raw_receiving_stats = pd.read_csv(self.raw_data_path + "/stats/wr_receiving_stats.csv")
        
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
            
        # Group by master player id and season
        filtered_receiving_stats = filtered_receiving_stats.groupby(['master_player_id', 'season']).agg({
            'receiving_yards': 'sum',
            **{col: 'first' for col in filtered_receiving_stats.columns if col not in ['master_player_id', 'season', 'receiving_yards']}
        }).reset_index()
        
        # Save to csv
        filtered_receiving_stats.to_csv(self.engineered_data_path + "/wr_receiving_stats.csv", index=False)
    
    def run(self):
        self.logger.info("Running Data Engineering Pipeline")
        
        self.players = pd.read_csv(self.raw_data_path + "/all_college_players.csv")
        
        # Engineer QB stats
        self.engineer_qbs()
        
        # Engineer RB stats
        self.engineer_rbs()
        
        # Engineer WR stats
        self.engineer_wrs()