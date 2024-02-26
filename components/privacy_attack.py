##### PLACE LICENSES HERE #####

import warnings
warnings.filterwarnings("ignore")

from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

##make sure this library works
import pykeops 
#make sure omp.h is also working, may need to google if warning occurs or run this code block twice
#make sure all libraries are include, you may need to find the header files and move them locally so they can be imported
pykeops.clean_pykeops()   
pykeops.test_numpy_bindings() 
pykeops.test_torch_bindings() 

# +++++++++++++++
# Anonymeter has multiple attack functions available
# We are using our modified version of Anonymeter
# https://github.com/statice/anonymeter
# +++++++++++++++
# NOTES:
#
#

from TODO_anon_inference_evaluator import InferenceEvaluator_Modified

# +++++++++++++++
# DOMIAS is a membership inference attack that focuses on overfitted distributions
# https://github.com/vanderschaarlab/DOMIAS
# +++++++++++++++
# NOTES:
#
#

from domias.evaluator import evaluate_performance
from domias.models.generator import GeneratorInterface



#PREPROCESSING METHODS
####
def categories_to_num(
    data: pd.DataFrame,
    columns: List[str]
):
    transformed_df = data
    for col in columns:
        uniques = transformed_df[col].unique()
        for index, value in enumerate(uniques):
            transformed_df.loc[transformed_df[col] == value, col] = index+1
    return transformed_df

# Drop Column
# used if there's a column that just is not working with the privacy attacks
def drop_column(
    data: pd.DataFrame,
    columns: List[str]    
):
    data.drop(
        columns, 
        axis='columns', 
        inplace=True
    )
    return

#PRIVACY ATTACK CLASS
####
# Available attacks:
# 1. Inference using Anonymeter and DOMIAS
####
class PrivacyAttack():

    def __init__(self):
        return
    def get_default_params(self):
        return {
            'anon_inf_attacks': 100,
            'domias_attacks': 100,
            'domias_mem_set_size': 0, #size of training data
            'domias_reference_set_size': 0, #size of control data
            'domias_synthetic_sizes': [1], #size of synthetic data
            'domias_density_estimator': "prior"  # prior, kde, bnaf
        }

    """ 
        Method: 
        Desc: 
        Out: 
    """
    def inference_attack(
        self,
        params: dict,
        original_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        control_data: Optional[pd.DataFrame] = None,
    ):
        # validate parameters
        # run anonymeter attack
        anon_results = self._anon_inference(
            original_data = original_data,
            synth_data = synth_data,
            control_data = control_data,
            n_attacks = params['anon_inf_attacks']
        )
        # run domias attack
        domias_results = self._domias_overfit(
            params = params,
            original_data = original_data,
            synth_data = synth_data,
            control_data = control_data
        )
        # combine results and return
        return {
            'anon_inference': anon_results,
            'domias': domias_results
        }
    
    # ===============
    #
    # PRIVATE METHODS
    #
    # ===============

    def _anon_inference(
        self,
        original_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        control_data: Optional[pd.DataFrame] = None,
        n_attacks: int = 500,
    ):
        if original_data.columns != synth_data.columns:
            return None
        
        columns = original_data.columns
        eval_results = []
        guesses = {}

        for secret in columns: 
            aux_cols = [col for col in columns if col != secret]
            # the attack algorithm uses the synthetic data to model the k-neighbors and then aux columns to guess secret column
            evaluator = InferenceEvaluator_Modified(ori=original_data, 
                                        syn=synth_data, 
                                        control=control_data,
                                        aux_cols=aux_cols,
                                        secret=secret,
                                        n_attacks=n_attacks)
            evaluator.evaluate(n_jobs=-2)
            

            # after modification of InferenceEvaluator to save the Guesses
            results = {
                'col': secret,
                'results': evaluator.results(),
                'y_pred': evaluator.guesses.to_list(), 
                'y_true': evaluator.targets[secret].to_list(),
                'y_distances': list(evaluator.distances)
            }

            eval_results.append(results)

        return eval_results

    def _get_generator(
        data
    ) -> GeneratorInterface:
        class LocalGenerator(GeneratorInterface):
            def __init__(self) -> None:
                self.data = data
            def fit(self, data: pd.DataFrame) -> "LocalGenerator":
                #do nothing, we already fit the data
                return self
            def generate(self) -> pd.DataFrame:
                return self.data
        return LocalGenerator()

    def _domias_overfit(
        self, 
        params: dict,
        original_data: pd.DataFrame,
        synth_data: pd.DataFrame
        # control_data: Optional[pd.DataFrame] = None
    ):
        # TODO: cut by param['size'] and combine original_data and control_data
        dataset = original_data.to_numpy()
        generator = self._get_generator(synth_data)
        perf = evaluate_performance(
            generator,
            dataset,
            params['domias_mem_set_size'],
            params['domias_reference_set_size'],
            training_epochs=1,
            synthetic_sizes=params['domias_synthetic_sizes'],
            density_estimator=params['domias_density_estimator'],
        )
        return perf