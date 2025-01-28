import pandas as pd

def get_team_division(espn_team_id: str, season: int) -> str:
    """
    Determine a team's division (G5, P5, or FCS) based on ESPN team ID.
    
    Args:
        espn_team_id (str): The ESPN team ID to look up
        filepath (str): Path to the CSV file containing team data
        
    Returns:
        str: The division (G5, P5, or FCS) of the team
        
    Raises:
        ValueError: If the team ID is not found in the data
    """
    try:
        # Read the CSV file
        filepath = '/util/helpers/teams.csv'
        df = pd.read_csv(filepath)
        
        # Find the most recent record for this team
        team_data = df[df['espn_team_id'] == espn_team_id]
        
        if team_data.empty:
            raise ValueError(f"No team found with ESPN ID: {espn_team_id}")
            
        # Get where season = season
        team_data = team_data[team_data['season'] == season]
        
        return team_data['division']
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find CSV file at: {filepath}")
    except Exception as e:
        raise Exception(f"Error processing team data: {str(e)}")


def load_team_divisions() -> pd.DataFrame:
    """
    Load team data from a CSV file.
    
    Returns:
        pd.DataFrame: The team data
    """
    try:
        # Read the CSV file with a different encoding
        filepath = './util/helpers/teams.csv'
        df = pd.read_csv(filepath, encoding='latin1')
        
        return df
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find CSV file at: {filepath}")
    except Exception as e:
        raise Exception(f"Error processing team data: {str(e)}")