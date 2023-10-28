from pydantic import BaseModel
from datetime import datetime
import pytz

# Fuseau horaire de Paris
paris_tz = pytz.timezone('Europe/Paris')

# Heure actuelle à Paris
current_time_paris = datetime.now(paris_tz)

# Heure et mois actuels
current_hour = current_time_paris.hour
current_month = current_time_paris.month

# Classe NewCall pour faire une prédiction.
class NewCall(BaseModel):
    HourOfCall: int = current_hour  #Time Automation
    IncGeo_BoroughCode: str
    IncGeo_WardCode: str
    Easting_rounded: int
    Northing_rounded: int
    IncidentStationGround: str
    NumStationsWithPumpsAttending: int
    NumPumpsAttending: int
    PumpCount: int
    PumpHoursRoundUp: int
    PumpOrder: int
    DelayCodeId: int
    Month: int = current_month  #Time Automation
    

