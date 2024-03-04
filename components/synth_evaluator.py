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
        self.metadata = metadata
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
        utility = self._gda_coverage(original_data, synth_data)
        #_marginal_distribution_similarity
        mds = self._marginal_distribution_similarity(original_data, synth_data)
        
        for column in original_data:
            utility[column]['mds'] = mds[column]

        return utility

    def run_data_diagnosis(self, original_data, synth_data):
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
        mds_scores = {}
        #build 2 lists using self.metadata: categorical columns and numerical columns
        categorical_cols = []
        numerical_cols = []
        metadata_dict = self.metadata.to_dict()
        for column in metadata_dict['columns']:
            if(metadata_dict['columns'][column]['sdtype'] == 'categorical'):
                categorical_cols.append(column)
            elif(metadata_dict['columns'][column]['sdtype'] == 'numerical'):
                numerical_cols.append(column)            

        #jsd
        for col in categorical_cols:
            # start by computing the probability of each value in both data (probability distribution)
            # probability = occurance / sample size
            counts_p = []
            counts_q = []
            uniques = original_data[col].unique()
            for value in uniques:
                counts_p.append(original_data[col].value_counts()[value])
                try:
                    counts_q.append(synth_data[col].value_counts()[value])
                except KeyError:
                    counts_q.append(0)

            prob_p = np.array(counts_p)/len(original_data[col])
            prob_q = np.array(counts_q)/len(synth_data[col])

            # now that we have our probability distributions, we use scipy's Jensen-Shannon distance and **2 to get divergence
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
            mds_scores[col] = jensenshannon(prob_p, prob_q) ** 2
        #kst
        for col in numerical_cols:
            mds_scores[col] = kstest(original_data[col], synth_data[col])
        return mds_scores

    def _gda_coverage(self, original_data, synth_data):
        # Coverage
        # We only need the original and synthetic datasets
        # https://github.com/gda-score/utility/blob/master/gdaUtility.py
        # For each column ... 
        # 1. line 261: build dicts for original and synthetic data counting all distinct values
        # 2. line 362: count values noColumnCountOnerawDb, noColumnCountMorerawDb, and valuesInBoth
        # 3. Coverage is calculated as valuesInBoth/noColumnCountMorerawDb
        coverage_scores = {}
        for col in original_data:
            # check if col is being covered
            if col not in synth_data:
                print("Not ", col)
                continue

        # line 230: if the column has continuous data, coverage = numAnonRows/numRawRows
            # print(metadata)
            if(self.metadata.columns[col]['sdtype'] == "numerical"):
                # print(col, " :TEST: ", metadata.columns[col]['sdtype'])
                entry = {}
                entry['column'] = col
                entry['coverage'] = synth_data[col].count()/original_data[col].count()
                coverage_scores[col] = entry
                continue

        # line 216: see how much of CATEGORICAL column is NULL
            # numRawRows = train_df[col].count() - train_df[col].value_counts()['?']
            # numAnonRows = synth_df[col].count() - synth_df[col].value_counts()['?']
            # if numRawRows == 0 or numAnonRows == 0:
            #     #empty column
            #     continue

        # line 250: count all distinct values in the column
        # line 261: build dicts for raw and anon
            rawRowsDict = {}
            anonRowsDict = {}
            for val in original_data[col].unique():
                if val == '?': continue
                rawRowsDict[val] = original_data[col].value_counts()[val]
            for val in synth_data[col].unique():    
                if val == '?': continue
                anonRowsDict[val] = synth_data[col].value_counts()[val]

        # line 362: count values
            noColumnCountOnerawDb=0
            noColumnCountMorerawDb=0
            valuesInBoth=0

            for rawkey in rawRowsDict:
                if rawRowsDict[rawkey]==1:
                    noColumnCountOnerawDb += 1
                else:
                    noColumnCountMorerawDb += 1
            for anonkey in anonRowsDict:
                if anonkey in rawRowsDict:
                    if rawRowsDict[anonkey] >1:
                        valuesInBoth += 1
            valuesanonDb=len(anonRowsDict)

            entry = {}
            entry['column'] = col
            entry['colCountOneRawDb']=noColumnCountOnerawDb
            entry['colCountManyRawDb']=noColumnCountMorerawDb
            entry['valuesInBothRawAndAnonDb']=valuesInBoth
            entry['totalValCntAnonDb']=valuesanonDb
            if(noColumnCountMorerawDb==0):
                entry['coverage'] =None
        # final calculation: coverage = valuesInBoth/noColumnCountMorerawDb
            else:
                entry['coverage']=valuesInBoth/noColumnCountMorerawDb

            coverage_scores[col] = entry

        return coverage_scores