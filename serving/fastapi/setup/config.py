import os
import sys
from typing import List, Optional, Union
from datetime import datetime

from pydantic import BaseModel, BaseSettings
from pydantic import Field
from pydantic import validator

from setup.handler import ModelHandler

class ProjectConfig(ModelHandler):
    def __init__(self, model_type='tde'):
        self.model_type = model_type
        self.threshold = 0.5
        self.project_path = os.path.abspath(os.getcwd())
        print(self.project_path)
        self.torch_model_path = f"{self.project_path}/torch_models"
        self.tensor_model_path = f"{self.project_path}/tensor_models"
        # self.add_example_path = "example/train_model"
        # self.model_path = f"{self.project_path}/models"
        self.tde_path = f"{self.torch_model_path}/dog/eye/"
        # self.nfm_path = f"{self.model_path}/nfm.pkl"
        # sys.path.append(f'./{self.add_example_path}')
        ModelHandler.__init__(self)


class VariableConfig:
    def __init__(self):
        self.host_list = ['127.0.0.1', '0.0.0.0']
        self.port_list = ['8000', '8088', '8080']


class APIEnvConfig(BaseSettings):
    host: str = Field(default='0.0.0.0', env='api host')
    port: int = Field(default='8000', env='api server port')
    
    # host 점검
    @validator("host", pre=True)
    def check_host(cls, host_input):
        if host_input == 'localhost':
            host_input = "127.0.0.1"
        if host_input not in VariableConfig().host_list:
            raise ValueError("host error")
        return host_input
    
    # port 점검
    @validator("port", pre=True)
    def check_port(cls, port_input):
        if port_input not in VariableConfig().port_list:
            raise ValueError("port error")
        return port_input


class APIConfig(BaseModel):
    api_name: str = 'main:app'
    api_info: APIEnvConfig = APIEnvConfig()


class DataInput(BaseModel):
    #user_id: int = Field(ge=0, le=1000)
    img_path:str    

class PredictOutput(BaseModel):
    prob:float
    prediction:int