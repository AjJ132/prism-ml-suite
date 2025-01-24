from logging import Logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import Generator
from contextlib import contextmanager
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
load_dotenv()


# Get environment variables with fallback values
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Verify all required environment variables are present
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise ValueError("Missing required database environment variables")

# Create the connection URL
DATABASE_URL = f"postgresql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Enable connection health checks
    pool_size=5,  # Adjust pool size based on your needs
    max_overflow=10
)

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_db(logger: Logger) -> Generator:
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        result = db.execute(text("SELECT current_database()")).scalar()
        logger.warning(f"Connected to database: {result}")
        
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        engine.dispose()
        db.close()