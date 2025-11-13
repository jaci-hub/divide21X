import datetime
import json

def get_utc_date(as_iso=True):
    """
    Returns the current UTC date.
    
    Parameters:
        as_iso (bool): If True, returns ISO string 'YYYY-MM-DD';
                       If False, returns a datetime.date object.
    
    Returns:
        str | datetime.date: UTC date in the desired format.
    """
    utc_dt = get_utc_datetime(as_iso=False)
    utc_date = utc_dt.date()
    return utc_date.isoformat() if as_iso else utc_date

def get_utc_datetime(as_iso=True):
    """
    Returns the current UTC datetime.
    
    Parameters:
        as_iso (bool): If True, returns ISO 8601 string; 
                       If False, returns a timezone-aware datetime object.
    
    Returns:
        str | datetime.datetime: UTC datetime in the desired format.
    """
    utc_dt = datetime.datetime.now(datetime.timezone.utc)
    return utc_dt.isoformat() if as_iso else utc_dt

def get_utc_day():
    """
    Returns the current day in UTC as an integer.
    """
    utc_dt = get_utc_datetime(as_iso=False)
    return utc_dt.day

def get_utc_hour():
    """
    Returns the current hour in UTC as an integer (0-23).
    """
    utc_dt = get_utc_datetime(as_iso=False)
    return utc_dt.hour

def get_llm_registry():
    '''
    returns the registry.json data
    '''
    # Load LLM registry
    registry = None
    try:
        with open("divide21x/llm_api/registry.json", "r") as f:
            registry = json.load(f)
    except Exception as e:
        registry = None
    
    return registry


if __name__ == "__main__":
    print(get_utc_day())