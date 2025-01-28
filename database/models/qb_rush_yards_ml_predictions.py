from sqlalchemy import MetaData, Table, Column, Integer, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
from typing import Dict, List
import pandas as pd

Base = declarative_base()
metadata = MetaData(schema='machine_learning')

class QbRushYdsMlPredictions(Base):
    __table__ = Table(
        'qb_rush_yds_ml_predictions',
        metadata,
        Column('id', Integer, primary_key=True),
        Column('master_player_id', UUID(as_uuid=True), ForeignKey('identity.players.player_id', ondelete='CASCADE'), nullable=False),
        Column('prediction_season', Integer, nullable=False),
        Column('previous_season_rush_yds', Integer),
        Column('ensemble_predicted_rush_yds', Integer),
        Column('gradient_boosting_predicted_rush_yds', Integer),
        Column('lasso_predicted_rush_yds', Integer),
        Column('lightgbm_predicted_rush_yds', Integer),
        Column('linear_regression_predicted_rush_yds', Integer),
        Column('neural_network_predicted_rush_yds', Integer),
        Column('random_forest_predicted_rush_yds', Integer),
        Column('ridge_predicted_rush_yds', Integer),
        Column('confidence_interval', Float),
        Column('uncertainty_percentage', Float),
        Column('date_created', DateTime, default=datetime.utcnow),
        Column('date_updated', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    )

    __table_args__ = (
        UniqueConstraint('master_player_id', 'prediction_season', name='qb_rush_yds_pred_uk'),
        {'schema': 'machine_learning'}
    )

    def __repr__(self):
        return f"<QbRushYdsMlPredictions(id={self.id}, master_player_id={self.master_player_id})>"

    @classmethod
    def upsert_prediction(cls, db_session, prediction_data: Dict) -> None:
        """
        Upsert a single prediction record.

        Args:
            db_session: SQLAlchemy session
            prediction_data: Dictionary containing prediction data fields
        """
        try:
            prediction_data['date_updated'] = datetime.utcnow()
            if 'date_created' not in prediction_data:
                prediction_data['date_created'] = datetime.utcnow()

            stmt = insert(cls.__table__).values(prediction_data)
            stmt = stmt.on_conflict_do_update(
                constraint='qb_rush_yds_pred_uk',
                set_=prediction_data
            )

            db_session.execute(stmt)
            db_session.commit()

        except Exception as e:
            db_session.rollback()
            raise Exception(f"Error upserting prediction: {str(e)}")

    @classmethod
    def bulk_upsert_predictions(cls, db_session, predictions_data: List[Dict]) -> None:
        """
        Bulk upsert multiple prediction records.

        Args:
            db_session: SQLAlchemy session
            predictions_data: List of dictionaries containing prediction data
        """
        try:
            current_time = datetime.utcnow()
            for prediction in predictions_data:
                prediction['date_updated'] = current_time
                if 'date_created' not in prediction:
                    prediction['date_created'] = current_time

            stmt = insert(cls.__table__)
            stmt = stmt.on_conflict_do_update(
                constraint='qb_rush_yds_pred_uk',
                set_={
                    'previous_season_rush_yds': stmt.excluded.previous_season_rush_yds,
                    'ensemble_predicted_rush_yds': stmt.excluded.ensemble_predicted_rush_yds,
                    'gradient_boosting_predicted_rush_yds': stmt.excluded.gradient_boosting_predicted_rush_yds,
                    'lasso_predicted_rush_yds': stmt.excluded.lasso_predicted_rush_yds,
                    'lightgbm_predicted_rush_yds': stmt.excluded.lightgbm_predicted_rush_yds,
                    'linear_regression_predicted_rush_yds': stmt.excluded.linear_regression_predicted_rush_yds,
                    'neural_network_predicted_rush_yds': stmt.excluded.neural_network_predicted_rush_yds,
                    'random_forest_predicted_rush_yds': stmt.excluded.random_forest_predicted_rush_yds,
                    'ridge_predicted_rush_yds': stmt.excluded.ridge_predicted_rush_yds,
                    'confidence_interval': stmt.excluded.confidence_interval,
                    'uncertainty_percentage': stmt.excluded.uncertainty_percentage,
                    'date_updated': current_time
                }
            )

            db_session.execute(stmt, predictions_data)
            db_session.commit()

        except Exception as e:
            db_session.rollback()
            raise Exception(f"Error bulk upserting predictions: {str(e)}")

    @classmethod
    def upsert_from_dataframe(cls, db_session, df: pd.DataFrame) -> None:
        """
        Upsert records from a pandas DataFrame.

        Args:
            db_session: SQLAlchemy session
            df: pandas DataFrame containing prediction data
        """
        try:
            predictions_data = df.to_dict('records')
            cls.bulk_upsert_predictions(db_session, predictions_data)
        except Exception as e:
            raise Exception(f"Error upserting from DataFrame: {str(e)}")
