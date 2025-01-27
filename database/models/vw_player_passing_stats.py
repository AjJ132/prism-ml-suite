import pandas as pd
from sqlalchemy import Table, Column, Integer, String, Float, DateTime, MetaData
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import List

Base = declarative_base()
metadata = MetaData()

class VwPlayerPassingStats(Base):
    __table__ = Table (
        'vw_player_passing_stats',
        metadata,
        Column('espn_player_id', String),
        Column('master_player_id', String, primary_key=True),
        Column('season', String, primary_key=True),
        Column('player_name', String),
        Column('position', String),
        Column('completions', String),
        Column('attempts', String),
        Column('passing_yards', String),
        Column('passing_tds', String),
        Column('interceptions', String),
        Column('qb_rating', String),
        Column('completion_percentage', String),
        Column('yards_per_attempt', String),
        Column('ensemble_predicted_passing_yards', String),
        Column('gradient_boosting_predicted_passing_yards', String),
        Column('lasso_predicted_passing_yards', String),
        Column('linear_regression_predicted_passing_yards', String),
        Column('neural_network_predicted_passing_yards', String),
        Column('random_forest_predicted_passing_yards', String),
        Column('ridge_predicted_passing_yards', String),
        Column('previous_season_passing_yards', String),
        Column('yards_vs_prediction', String),
        Column('prediction_accuracy_percentage', String),
        schema='views'
    )
    
    def __repr__(self):
        return f"<VwPlayersPassingStats(master_player_id={self.master_player_id}, name={self.name})>"
    
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
    def fetch_player_passing_stats(cls, db_session) -> pd.DataFrame:
        """
        Fetch passing stats for players

        Args:
            db_session (_type_): _description_
        """
        
        query = db_session.query(cls)
        df = pd.read_sql(query.statement, db_session.bind)
        
        # Convert timestamp columns to datetime
        timestamp_columns = ['created_at', 'updated_at']
        for col in timestamp_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        return df