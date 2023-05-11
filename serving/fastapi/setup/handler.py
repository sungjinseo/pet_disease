import torch

class ModelHandler:
    def __init__(self):
        # gpu 쓴다면 바꿔야함
        self.device = torch.device('cpu')
        self.model_file_name = 'checkpoint.pt'
        self.label_file_name = 'classes.txt'
    

    def load_model(self):
        if self.model_type == 'tde':
            model = torch.load(f'{self.tde_path}'+ self.model_file_name, map_location=self.device)
        else:
            model = torch.load(f'{self.tde_path}'+ self.model_file_name, map_location=self.device)
        return model
    
    label_file_name = 'classes.txt'

    def load_label(self):
        with open(f'{self.tde_path}'+ self.label_file_name) as f:
            return [line.strip() for line in f.readlines()]

class DataHandler:
    def check_type(self, check_class, data):
        data = check_class(**data)
        return data