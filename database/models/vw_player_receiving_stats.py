import pandas as pd
from sqlalchemy import Table, Column, Integer, String, Float, DateTime, MetaData
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import List

Base = declarative_base()
metadata = MetaData()

class VwPlayerReceivingStats(Base):
    __table__ = Table(
        'vw_player_receiving_stats',
        metadata,
        Column('espn_player_id', String),
        Column('master_player_id', String, primary_key=True),
        Column('season', String, primary_key=True),
        Column('player_name', String),
        Column('position', String),
        Column('receptions', String),
        Column('receiving_yards', String),
        Column('receiving_tds', String),
        Column('long', String),
        Column('yards_per_reception', String),
        Column('ensemble_predicted_receiving_yds', String),
        Column('gradient_boosting_predicted_receiving_yds', String),
        Column('lasso_predicted_receiving_yds', String),
        Column('linear_regression_predicted_receiving_yds', String),
        Column('neural_network_predicted_receiving_yds', String),
        Column('random_forest_predicted_receiving_yds', String),
        Column('ridge_predicted_receiving_yds', String),
        Column('previous_season_receiving_yds', String),
        Column('yards_vs_prediction', String),
        Column('prediction_accuracy_percentage', String),
        schema='views'
    )
    
    def __repr__(self):
        return f"<VwPlayersReceivingStats(master_player_id={self.master_player_id}, name={self.name})>"
    
    @classmethod
    def to_dataframe(cls, db_session, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch players from the view and return as a pandas DataFrame.
        
        Args:
            db_session: SQLAlchemy session
            limit: Maximum number of records to return
            
        Returns:
            pandas DataFrame containing player data
        """
        # Execute query and fetch all results
        query = db_session.query(cls).limit(limit)
        
        # Convert to DataFrame
        df = pd.read_sql(query.statement, db_session.bind)
        
        # Optional: Convert timestamp columns to datetime if needed
        timestamp_columns = ['created_at', 'updated_at']
        for col in timestamp_columns:
            if (col in df.columns):
                df[col] = pd.to_datetime(df[col])
                
        return df
    
    @classmethod
    def to_filtered_dataframe(cls, db_session, 
                            filters: dict = None, 
                            limit: int = 1000) -> pd.DataFrame:
        """
        Fetch filtered players data and return as a pandas DataFrame.
        
        Args:
            db_session: SQLAlchemy session
            filters: Dictionary of column:value pairs to filter by
            limit: Maximum number of records to return
            
        Returns:
            pandas DataFrame containing filtered player data
        """
        query = db_session.query(cls)
        
        if filters:
            for column, value in filters.items():
                if hasattr(cls.__table__.c, column):
                    query = query.filter(getattr(cls.__table__.c, column) == value)
        
        query = query.limit(limit)
        df = pd.read_sql(query.statement, db_session.bind)
        
        # Convert timestamp columns to datetime
        timestamp_columns = ['created_at', 'updated_at']
        for col in timestamp_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        return df
    
    @classmethod
    def fetch_player_receiving_stats(cls, db_session) -> pd.DataFrame:
        """
        Fetch receiving stats for players

        Args:
            db_session: SQLAlchemy session for database connection
            
        Returns:
            pandas DataFrame containing player receiving stats
        """
        query = db_session.query(cls)
        df = pd.read_sql(query.statement, db_session.bind)
        
        # Convert timestamp columns to datetime
        timestamp_columns = ['created_at', 'updated_at']
        for col in timestamp_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        return df