from spn.io.Text import str_to_spn
from spn.algorithms.Inference import likelihood
import numpy as np
import tensorflow as tf
import RAT_SPN
import region_graph
import model
from main import Config, CspnTrainer
from main import MnistDataset

class CausalSPN():

    def __init__(self, models_cspn, models_spn):

        self.models_spn = {}
        for i, k in enumerate(models_spn):
            self.models_spn.update({k: self.load_spn_model(models_spn[k])})

        self.models_cspn = {}
        for i, k in enumerate(models_cspn):
            self.models_cspn.update({k: self.load_cspn_model(models_cspn[k])})

    def load_spn_model(self, model_spn_path):

        with open(model_spn_path, 'r') as f:
            str_spn = f.readlines()
        str_spn = ''.join(str_spn)

        return str_to_spn(str_spn)

    def load_cspn_model(self, model_cspn_path):

        return ...

    def p(self, query):

        for k in query:
            ...