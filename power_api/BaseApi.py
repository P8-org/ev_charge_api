
from pydantic import BaseModel

# abstracted in case more api calls are needed
# need better name
class BaseApiCall(BaseModel):
    def call_api(self):
        pass
    def get_period(self):
        pass
