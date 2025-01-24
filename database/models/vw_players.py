import pandas as pd
from sqlalchemy import Table, Column, Integer, String, Float, DateTime, MetaData
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import List

Base = declarative_base()
metadata = MetaData()

class VwPlayers(Base):
    __table__ = Table(
        'vw_players',
        metadata,
        Column('master_player_id', Integer, primary_key=True),
        Column('player_source_id', String),
        Column('ap_team_id', Integer),
        Column('team_name', String),
        Column('team_image_url', String),
        Column('name', String),
        Column('position', String),
        Column('class', String),
        Column('height_inches', Integer),
        Column('weight_lbs', Integer),
        Column('birthplace', String),
        Column('profile_img_url', String),
        Column('inferred_season', String),
        Column('created_at', DateTime),
        Column('updated_at', DateTime),
        Column('school_type', String),
        Column('source', String),
        Column('valuation', String),
        Column('valuation_number', Float),
        Column('nil_rating', Float),
        Column('instagram_url', String),
        Column('instagram_followers', Integer),
        Column('x_url', String),
        Column('x_followers', Integer),
        Column('tiktok_url', String),
        Column('tiktok_followers', Integer),
        schema='views'  # Specify the schema
    )
    
    def __repr__(self):
        return f"<VwPlayers(master_player_id={self.master_player_id}, name={self.name})>"
    
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
    
    # Example usage with your existing database connection:
    """
    with get_db() as db:
        players = fetch_players(db)
        for player in players:
            print(f"Player: {player.name}, Position: {player.position}")
    """
    # fetch all players (with limit)
    @classmethod
    def fetch_players(cls, db_session, limit: int = 1000):
        """
        Fetch players from the view with an optional limit.
        
        Args:
            db_session: SQLAlchemy session
            limit: Maximum number of records to return
            
        Returns:
            List of VwPlayers objects
        """
        return db_session.query(cls).limit(limit).all()

    @classmethod
    def fetch_all_college_players(cls, db_session) -> pd.DataFrame:
        """
        Fetch all players where source = 'espn' as a DataFrame.

        Args:
            db_session: SQLAlchemy session
            
        Returns:
            pandas DataFrame containing college player data
        """
        query = db_session.query(cls).filter(cls.source == 'espn')
        df = pd.read_sql(query.statement, db_session.bind)
        
        # Convert timestamp columns to datetime
        timestamp_columns = ['created_at', 'updated_at']
        for col in timestamp_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        return df
        
        