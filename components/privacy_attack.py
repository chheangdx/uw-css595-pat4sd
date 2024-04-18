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

from .anon_inference_evaluator import InferenceEvaluator_Modified

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

# Drop Column
# used if there's a column that just is not working with the privacy attacks
def drop_columns(
    data: pd.DataFrame,
    columns: List[str]    
):
    transformed_df = data
    transformed_df.drop(
        columns, 
        axis='columns', 
        inplace=True
    )
    return transformed_df

#PRIVACY ATTACK CLASS
####
# Available attacks:
# 1. Inference using Anonymeter and DOMIAS
####
class PrivacyAttack():

    def __init__(self, metadata):
        self.metadata = metadata
        return
    def get_default_params(self):
        return {
            'anon_inf_attacks': 100,
            'domias_mem_set_size': 1, #size of training data
            'domias_reference_set_size': 0, #size of control data
            'domias_synthetic_sizes': 1, #size of synthetic data
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
        # TODO: validate parameters

        # run anonymeter attack
        print("Running Anon Attack")
        anon_results = self._anon_inference(
            original_data = original_data,
            synth_data = synth_data,
            control_data = control_data,
            n_attacks = params['anon_inf_attacks']
        )
        # run domias attack
        print("Running Domias Attack")
        domias_perf = self._domias_overfit(
            params = params,
            original_data = original_data,
            synth_data = synth_data,
            control_data = control_data
        )
        domias_results = self._convert_domias_perf_to_results(
            perf = domias_perf,
            params = params
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
        n_attacks: int = 500
    ):
        eval_results = {}
        for secret in original_data.columns: 
            aux_cols = [col for col in original_data.columns if col != secret]
            # the attack algorithm uses the synthetic data to model the k-neighbors and then aux columns to guess secret column
            evaluator = InferenceEvaluator_Modified(ori=original_data, 
                                        syn=synth_data, 
                                        control=control_data,
                                        aux_cols=aux_cols,
                                        secret=secret,
                                        n_attacks=n_attacks)
            evaluator.evaluate(n_jobs=-2)
            
            print(secret," - Attack Completed, Processing Results")
            # after modification of InferenceEvaluator to save the Guesses
            results = {
                # https://github.com/statice/anonymeter/blob/3a7408156b572de67a01277ce50bf485bd7c9529/src/anonymeter/stats/confidence.py#L168
                'col': secret,
                'results': evaluator.results(),
                'guesses': evaluator.guesses.to_list(), 
                'targets': evaluator.targets[secret].to_list(),
                'distances': list(evaluator.distances)
            }
            y_true = []
            y_pred = []
            for i in range(0, len(results['targets'])):
                y_pred.append(1)
                if results['targets'][i] == results['guesses'][i]:
                    y_true.append(1)
                else:
                    y_true.append(0)
                
            results['y_true'] = y_true
            results['y_pred'] = y_pred

            eval_results[secret] = results

        return eval_results

    def _get_generator(
        self,
        data
    ) -> GeneratorInterface:
        class LocalGenerator(GeneratorInterface):
            def __init__(self) -> None:
                self.data = data
            def fit(self, data: pd.DataFrame) -> "LocalGenerator":
                #do nothing, we already fit the data
                return self
            def generate(self, count: int) -> pd.DataFrame:
                if(count < self.data.shape[0]):
                    return self.data.iloc[0:count]
                else:
                    return self.data
        return LocalGenerator()

    def _domias_overfit(
        self, 
        params: dict,
        original_data: pd.DataFrame,
        synth_data: pd.DataFrame,
        control_data: Optional[pd.DataFrame] = None
    ):
        original_df = pd.concat([original_data, control_data])
        synth_df = synth_data
        # do data conversion here
        cat_columns = []
        metadata_dict = self.metadata.to_dict()
        for column in metadata_dict['columns']:
            if(metadata_dict['columns'][column]['sdtype'] == 'categorical'):
                cat_columns.append(column)
        if(len(cat_columns) > 0):
            transformed_dfs = self._categories_to_num(
                datasets_to_convert = [original_df, synth_data],
                columns = cat_columns
            )
            del original_df
            del synth_df
            original_df = transformed_dfs[0]
            synth_df = transformed_dfs[1]

        dataset = original_df.to_numpy()
        del original_df
        generator = self._get_generator(synth_df)
        print("Attack Parameters Process, Attacking")
        perf = evaluate_performance(
            generator,
            dataset,
            params['domias_mem_set_size'],
            params['domias_reference_set_size'],
            training_epochs=1,
            synthetic_sizes=[params['domias_synthetic_sizes']],
            density_estimator=params['domias_density_estimator'],
        )
        print("Attack Completed, Processing Results")
        return perf
    
    def _convert_domias_perf_to_results(
        self, 
        perf, 
        params
    ):
        results = perf[params['domias_synthetic_sizes']]['data']
        y_true = results['Ytest']
        mia_scores = perf[params['domias_synthetic_sizes']]['MIA_scores']['domias']
        y_pred = mia_scores > np.median(mia_scores)
        n_attacks = y_pred.size
        n_success = (y_pred == True).sum()

        y_true_binary = []
        y_pred_binary = []
        y_success_binary = []
        for i in range(0, len(mia_scores)):
            y_true_binary.append(1 if y_true[i] else 0)
            y_pred_binary.append(1 if y_pred[i] else 0)
            if y_true_binary[i] == 1 and y_pred_binary[i] == 1:
                y_success_binary.append(1)
            else:
                y_success_binary.append(0)

        return {
            'y_true': y_true_binary,
            'y_pred': y_pred_binary,
            'y_success': y_success_binary,
            'mia_scores': mia_scores,
            'n_attacks': n_attacks,
            'n_success': n_success
        }
   
    # TODO??: make a separate method for capturing uniques and this method should take that unique list as a param
    def _categories_to_num(
        self,
        datasets_to_convert: List[pd.DataFrame],
        columns: List[str],

    ):
         # to keep consistency between multiple dataframes, we use a reference and then convert everything else
        reference_data = pd.concat(datasets_to_convert)
        transformed_dfs = datasets_to_convert
        for col in columns:
            uniques = reference_data[col].unique()
            for index, value in enumerate(uniques):
                for dataset in transformed_dfs:
                    dataset.loc[dataset[col] == value, col] = index+1
            # need to make sure everything is numeric
            for dataset in transformed_dfs:
                dataset[col] = pd.to_numeric(dataset[col])
            
        return transformed_dfs