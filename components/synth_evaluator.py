##### PLACE LICENSES HERE #####

import warnings
warnings.filterwarnings("ignore")

# IMPORTS AND DATA
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

#for Anonymeter
from scipy.stats import norm
from math import sqrt

from scipy.spatial.distance import jensenshannon
from scipy.stats import kstest

#for DOMIAS
from sklearn.metrics import accuracy_score, roc_auc_score

#for GDA
from statistics import mean, stdev

#for SDV
import sdv
from sdv.metadata import SingleTableMetadata
import graphviz
from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic
from sdv.evaluation.single_table import get_column_plot
from sdv.evaluation.single_table import get_column_pair_plot

#RISK EVALUATION CLASS
####
class SynthEvaluator():

    def __init__(self, metadata):
        return

    """ 
        Method: 
        Desc: 
        Out: 
    """
    def run_defense(self):
        #_attack_accuracy
        #_gda_defense
        return

    """ 
        Method: 
        Desc: 
        Out: 
    """
    def run_utility(self, original_data, synth_data):
        #_gda_coverage
        #_marginal_distribution_similarity
        return

    def run_data_diagnosis(self, original_data, synth_data, test_data=None):
        print("=== Quality Report ===")
        quality_report = evaluate_quality(
            real_data=original_data,
            synthetic_data=synth_data,
            metadata=self.metadata
        )

        print("=== Diagnostic Report ===")
        diagnostic_report = run_diagnostic(
            real_data=original_data,
            synthetic_data=synth_data,
            metadata=self.metadata
        )

        return

    def run_column_diagnosis(self, original_data, synth_data, column):
        fig = get_column_plot(
            real_data=original_data,
            synthetic_data=synth_data,
            column_name=column,
            metadata=self.metadata
        )
        fig.show()
        return
    
    def run_two_columns_diagnosis(self, original_data, synth_data, column1, column2):
        fig = get_column_pair_plot(
            real_data=original_data,
            synthetic_data=synth_data,
            metadata=self.metadata,
            column_names=[column1, column2],
        )
        fig.show()
        return
    # ===============
    #
    # PRIVATE METHODS
    #
    # ===============

    def _attack_accuracy(self):
        #anon version

        #domias version

        #rocauc

        #GDA version (modified from Utility score)

        return

    def _gda_defense(self):
        
        return

    def _marginal_distribution_similarity(self, original_data, synth_data):
        
        #build 2 lists using self.metadata: categorical columns and numerical columns

        #jsd

        #kst
        
        return

    def _gda_coverage(self):

        return