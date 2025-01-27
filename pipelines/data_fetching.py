


from logging import Logger
from database.db_manager import get_db
from database.models.vw_player_passing_stats import VwPlayerPassingStats
from database.models.vw_player_rushing_stats import VwPlayerRushingStats
from database.models.vw_players import VwPlayers
import pandas as pd


class DataFetchingPipeline:
    def __init__(self, logger: Logger, valid_position, mode: int, raw_data_path: str = "data/raw"):
        self.logger = logger
        self.valid_position = valid_position
        self.mode = mode
        self.raw_data_path = raw_data_path
        self.players = None
        
    def fetch_college_players(self):
        self.logger.info("Fetching College Players")
        
        with get_db(logger=self.logger) as db:
            df = VwPlayers.fetch_all_college_players(db)
            
            self.logger.info(f"Fetched {len(df)} college players")
            
            #sort by name
            df = df.sort_values(by='name')
            
            # Save to CSV
            df.to_csv(self.raw_data_path + "/all_college_players.csv", index=False)
            
            #break up into positions
            for position in self.valid_position:
                df_position = df[df['position'] == position]
                df_position.to_csv(self.raw_data_path + f"/by_position/college_players_{position}.csv", index=False)
                
            #get all unique player ids
            self.players = df
                
    def fetch_qb_stats(self,):
        self.logger.info("Fetching QB Stats")
        
        qbs = self.players.copy()
        
        #limit to quarterbacks
        qbs = qbs[qbs['position'] == 'QB']
        
        player_ids = qbs['master_player_id'].unique()
        
        self.logger.info(f"Fetching stats for {len(player_ids)} Quarterbacks")
        
        #fetch all passing stats
        with get_db(logger=self.logger) as db:
            passing_stats = VwPlayerPassingStats.fetch_player_passing_stats(db)
            
            self.logger.info(f"Found {len(passing_stats)} unfiltered passing stats records")
            
            #limit to where position = QB
            passing_stats = passing_stats[passing_stats['position'] == 'QB']
            self.logger.info(f"{len(passing_stats)} after position filtering")        
            
            #save to csv
            passing_stats.to_csv(self.raw_data_path + f"/stats/qb_passing_stats.csv", index=False)
            
    def fetch_rb_stats(self,):
        self.logger.info("Fetching QB Stats")
        
        rbs = self.players.copy()
        
        #limit to quarterbacks
        rbs = rbs[rbs['position'] == 'RB']
        
        player_ids = rbs['master_player_id'].unique()
        
        self.logger.info(f"Fetching stats for {len(player_ids)} Runningbacks")
        
        #fetch all passing stats
        with get_db(logger=self.logger) as db:
            passing_stats = VwPlayerRushingStats.fetch_player_rush_stats(db)
            
            self.logger.info(f"Found {len(passing_stats)} unfiltered rushing stats records")
            
            #limit to where position = QB
            passing_stats = passing_stats[passing_stats['position'] == 'RB']
            self.logger.info(f"{len(passing_stats)} after position filtering")        
            
            #save to csv
            passing_stats.to_csv(self.raw_data_path + f"/stats/rb_passing_stats.csv", index=False)
        
            

    def run(self):
        print('\n')
        self.logger.info("Begining Data Fetching Pipeline")
        
        # Fetch college players
        self.fetch_college_players()
        
        #Fetch Quarterback stats
        self.fetch_qb_stats()
        #Fetch Runningback stats
        self.fetch_rb_stats()
        #Fetch Receviver stats
        
        