


from logging import Logger
from database.db_manager import get_db
from database.models.vw_players import VwPlayers
import pandas as pd


class DataFetchingPipeline:
    def __init__(self, logger: Logger, valid_position, mode: int, raw_data_path: str = "data/raw"):
        self.logger = logger
        self.valid_position = valid_position
        self.mode = mode
        self.raw_data_path = raw_data_path
        
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

    def run(self):
        self.logger.info("Begining Data Fetching Pipeline")
        
        # Fetch college players
        self.fetch_college_players()
        
        