import datetime
from pydantic import BaseModel

# abstracted in case more api calls are needed

class RequestDetail(BaseModel):
    startDate: datetime.date
    endDate: datetime.date

# need better name
class BaseApiCall(BaseModel):
    def __init__(self):
        pass
    def call_api(self, rd: RequestDetail): # brug: if not insteance(rd, NY TYPE NAVN)
        pass
    def get_period(self):
        pass


