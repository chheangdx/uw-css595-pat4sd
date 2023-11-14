import pandas as pd
import graphviz

# +++++++++++++++
# SDV is used for generating CTGAN
# https://docs.sdv.dev/sdv/
# +++++++++++++++
import sdv
# load each sdv module that we support
from sdv.single_table.base import BaseSingleTableSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata

# +++++++++++++++
# SynthNoise is used for DP-CTGAN
# https://github.com/opendp/smartnoise-sdk/tree/main/synth
# +++++++++++++++
from snsynth import Synthesizer



class DataSynthesis:
    data_synthesis_approaches = ['ctgan', 'dpctgan']

    # metadata for the dataset to be used must be provided
    # metadata can be created during preprocessing
    def __init__(self, metadata):
        self.metadata = metadata

    """ 
        Method: synth_data 
        Desc: Takes a dataset and runs data synthesis method on it
        Assumptions: For simplicity (not efficiency), the trained model is never saved nor loaded
        In:
            data: (dataframe) training dataset, the dataset is assumed to already have its necessary preprocessing
            approach: (string) approach for synthesis, see get_approaches for options
            parameters: (dict) for any modifications to the parameters of the approach
                see get_default_params for available parameters to modify
                if no changes to parameters, None will run the approach with defaults
        Out: synthetic dataset as a dataframe
    """
    def synth_data(self, data, approach='ctgan', parameters=None):
        if approach == 'ctgan':
            return self._run_sdv(data, approach, parameters)
        if approach == 'dpctgan':
            return self._run_sn(data, approach, parameters)
        else:
            return None

    """ 
        Method: get_approaches 
        Desc: lists available approaches
        Out: approaches listed by data_synthesis_approaches
    """
    def get_approaches(self):
        return self.data_synthesis_approaches
    """ 
        Method: get_default_params 
        Desc: provides default parameters for a specified approach
        In:
            approach: (string) approach for synthesis, see get_approaches for options
        Out: dict with parameter names and default values
    """
    def get_default_params(approach):
        if approach == 'ctgan':
            return {
                'sample_size': 1000,
                'enforce_round': False,
                'epochs': 500,
                'verbose': True
            }
        if approach == 'dpctgan':
            return {
                'sample_size': 1000,
                'generator_decay': (10**-5),
                'discriminator_decay': (10**-3),
                'batch_size': 64,
                'epochs': 100,
                'epsilon': 32,
                'verbose': True,
                'preprocessor_eps': 1.0
            }
        else:
            return None
    """ 
        Method: run_data_analysis 
        Desc: Analyzes synthetic data to its train data and test data
        In: 
            train_data: (dataframe) original data used for training
            synth_data: (dataframe) synthetic data that was created from trained ML model
            test_data: (dataframe) (optional) original data that wasnt used for training
        Out: 0, analysis data is printed here.
    """
    def run_data_analysis(self, train_data, synth_data, test_data=None):
        return 0
    
    # ===============
    #
    # PRIVATE METHODS
    #
    # ===============
    
    def _run_sdv(self, data, approach, parameters):
        params = parameters
        if params is None:
            params = self.get_default_params('ctgan')
        
        synthesizer = BaseSingleTableSynthesizer(self.metadata)
        if approach == 'ctgan':
            synthesizer = CTGANSynthesizer(
                metadata=self.metadata,
                enforce_rounding=params['enforce_rounding'],
                epochs=params['epochs'],
                verbose=params['verbose']
            )
            synthesizer.fit(data)
        # TODO: Other methods are tvae, gaussiancopula, copulagan
        synthetic_data = synthesizer.sample(num_rows=params['sample_size'])
        return 0
    
    def _run_sn(self, data, approach, parameters):
        params = parameters
        if params is None:
            params = self.get_default_params('dpctgan')
        synth = Synthesizer.create(
            approach, 
            generator_decay = params['generator_decay'],
            discriminator_decay = params['discriminator_decay'], 
            batch_size = params['batch_size'], 
            epochs = params['epochs'], 
            epsilon = params['epsilon'], 
            verbose = params['verbose']
        )
        synth.fit(data, preprocessor_eps=params['preprocessor_eps'])
        dataframe_synth = synth.sample(params['sample_size'])

        return 0
