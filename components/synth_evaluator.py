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
    def run_defense(self, attack_results):
        score = {}
        score['accuracy'] = self._attack_accuracy(attack_results)
        score['pairwise_error'] = self._gda_pairwise_error(attack_results)
        score['gda_defense'] = self._gda_defense(attack_results)
        return score

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

    def _attack_accuracy(self, attack_results):
        confidence_level = 0.95
        accuracy = {}
        accuracy['anon_inference'] = {}
        accuracy['domias'] = {}
        for col_result in attack_results['anon_inference'].values():
            #anonymeter version of accuracy calculation
            n_attacks = col_result['results'].n_attacks
            n_success = col_result['results'].n_success
            z = norm.ppf(0.5 * (1.0 + confidence_level))
            z_squared = z * z
            n_success_var = n_success * (n_attacks - n_success) / n_attacks
            denominator = n_attacks + z_squared
            rate1 = (n_success + 0.5 * z_squared) / denominator
            error1 = (z / denominator) * sqrt(n_success_var + 0.25 * z_squared)
            #domias version of accuracy calculation
            rate2  = accuracy_score(col_result['y_true'], col_result['y_pred'])
            accuracy['anon_inference'][col_result['col']] = {
                'rate1': rate1,
                'error1': error1,
                'rate2': rate2
            }
            # TODO add control and baseline attack results
            
        #then do it for DOMIAS results
        #anonymeter version of accuracy calculation
        n_attacks = attack_results['domias']['n_attacks']
        n_success = attack_results['domias']['n_success']
        z = norm.ppf(0.5 * (1.0 + confidence_level))
        z_squared = z * z
        n_success_var = n_success * (n_attacks - n_success) / n_attacks
        denominator = n_attacks + z_squared
        rate1 = (n_success + 0.5 * z_squared) / denominator
        error1 = (z / denominator) * sqrt(n_success_var + 0.25 * z_squared)
        #domias version of accuracy calculation
        rate2  = accuracy_score(attack_results['domias']['y_true'], attack_results['domias']['y_pred'])
        accuracy['domias'] = {
            'rate1': rate1,
            'error1': error1,
            'rate2': rate2
        }
        #rocauc
        # AUCROC as defined by DOMIAS
        # use sklearn's roc_auc_score function, which uses y_true and y_scores
        # for Anonymeter, our y_scores would be the distance measured
        for col_result in attack_results['anon_inference'].values():
            try:
                rocauc = roc_auc_score(col_result['y_true'], col_result['distances'])
                accuracy['anon_inference'][col_result['col']]['rocauc'] = rocauc
            except ValueError:
                print(col_result['col'], ' could not calculate ROCAUC due to single class in y_true')
                pass
        # aucroc also is measured on continuous datasets, so this may not fit
        domias_rocauc = roc_auc_score(attack_results['domias']['y_true'], attack_results['domias']['mia_scores'])
        accuracy['domias']['rocauc'] = rocauc
        return accuracy

    def _gda_pairwise_error(self, attack_results):
        # We can also use the Accuracy algorithm from GDA's utility score to further measure attack accuracy
        # 1. Create Error Lists:
        #     ** skip all Anon where 0 before computing, not needed here because anon is all 1
        #     Absolute Error: Absolute( Anon - Raw)
        #     Simple Relative Error: Raw / Anon
        #     Relative Error: Absolute(Anon - Raw) / Max (Anon, Raw)
        # 2. Convert Error Lists into 5 metrics each: Min, Max, Avg, Stddev, Compute
        # print("::Anonymeter Results::")
        gda_acc = {}
        gda_acc['anon_inference'] = {}
        for col_result in attack_results['anon_inference'].values():
            absErrorList = []
            simpleRelErrorList = []
            relErrorList = []

            y_true = col_result['y_true']
            y_pred = col_result['y_pred']
            for i in range(0, len(y_true)):
                absErrorList.append(abs(y_pred[i]-y_true[i]))
                simpleRelErrorList.append(y_true[i]/y_pred[i])
                relErrorList.append(abs(y_pred[i]-y_true[i]) / max(y_pred[i], y_true[i]))
            mins = [min(absErrorList), min(simpleRelErrorList), min(relErrorList)]
            maxs = [max(absErrorList), max(simpleRelErrorList), max(relErrorList)]
            avgs = [mean(absErrorList), mean(simpleRelErrorList), mean(relErrorList)]
            stdevs = [
                round(stdev(absErrorList), 5), 
                round(stdev(simpleRelErrorList), 5), 
                round(stdev(relErrorList), 5)
            ]
            
            mse_result = []
            for errorList in [absErrorList, simpleRelErrorList, relErrorList]:
                mse = 0
                for item in errorList:
                    mse += item * item
                if len(errorList) > 0:
                    mse = mse/len(errorList)
                mse_result.append(mse)
            
            col = col_result['col']
            gda_acc['anon_inference'][col] = {}
            
            #our GDA calculated accuracy score
            gda_acc['anon_inference'][col]['mins'] = mins
            gda_acc['anon_inference'][col]['maxs'] = maxs
            gda_acc['anon_inference'][col]['avgs'] = avgs
            gda_acc['anon_inference'][col]['stdevs'] = stdevs
            gda_acc['anon_inference'][col]['mse'] = mse_result
            # print(str(col).ljust(15, ' '), " --> ", gda_acc[col])

        # print("\n\n::DOMIAS Results::")
        absErrorList = []
        simpleRelErrorList = []
        relErrorList = []

        y_true = attack_results['domias']['y_true']
        y_pred = attack_results['domias']['y_pred']
        for i in range(0, len(y_true)):
            #skip 0s
            if(y_pred[i] == 0):
                continue
            absErrorList.append(abs(y_pred[i]-y_true[i]))
            simpleRelErrorList.append(y_true[i]/y_pred[i])
            relErrorList.append(abs(y_pred[i]-y_true[i]) / max(y_pred[i], y_true[i]))
        mins = [min(absErrorList), min(simpleRelErrorList), min(relErrorList)]
        maxs = [max(absErrorList), max(simpleRelErrorList), max(relErrorList)]
        avgs = [mean(absErrorList), mean(simpleRelErrorList), mean(relErrorList)]
        stdevs = [
            round(stdev(absErrorList), 5), 
            round(stdev(simpleRelErrorList), 5), 
            round(stdev(relErrorList), 5)
        ]
        mse_result = []
        for errorList in [absErrorList, simpleRelErrorList, relErrorList]:
            mse = 0
            for item in errorList:
                mse += item * item
            if len(errorList) > 0:
                mse = mse/len(errorList)
            mse_result.append(mse)

        #our GDA calculated accuracy score
        gda_acc['domias'] = {}
        gda_acc['domias']['mins'] = mins
        gda_acc['domias']['maxs'] = maxs
        gda_acc['domias']['avgs'] = avgs
        gda_acc['domias']['stdevs'] = stdevs
        gda_acc['domias']['mse'] = mse_result
        return gda_acc

    def _gda_defense(self, attack_results):
        # GDA Scores - Defense
        # unable to do: requires guesses and confidence of each guess
        # Anonymeter can provide guesses if we rewrite the attack code ourselves, but we won't get confidence
        # DOMIAS can provide guesses and something similar to confidence (MIA Scores), but it is also different because the "guesses" is all the original data

        # Confidence Improvement
        # For Anonymeter attack: CI = (C-S)/(1-S) where C=n_success/n_attacks and S = n_baseline/n_attacks
        # For DOMIAS attack: C=n_success/n_attacks and S=count(True)/total? Or maybe we use Anonymeter S as the baseline confidence
        Sum_S = 0
        Count_S = 0
        anon_CICP = {}
        for col_result in attack_results['anon_inference'].values():
            anon_n_attacks = col_result['results'].n_attacks
            anon_n_success = col_result['results'].n_success
            anon_n_baseline = col_result['results'].n_baseline

            C_anon = anon_n_success/anon_n_attacks
            S_anon = anon_n_baseline/anon_n_attacks
            Sum_S+= S_anon
            Count_S+=1
            # assume anonymter baseline attack is applicable for both anonymter and domias
            ## S_domias = sum(domias_y_success)/(len(domias_y_success)+1)
            CI_anon = (C_anon-S_anon)/(1-S_anon)
            # Defense & Confidence are basically the same in our interpretation
            # Using defense gride, getInterpolatedValue(CI, CP, defenseGrid)
            # Claim made is most likely how many claims met the confidence threshold, which we do not have
            # CP is defined as the ratio of attempts to claims
            # CP is calculated as claim made / claim trials, but we can try success/total
            CP_anon = anon_n_success/anon_n_attacks

            anon_CICP[col_result['col']] = {}
            anon_CICP[col_result['col']]['ci'] = CI_anon
            anon_CICP[col_result['col']]['cp'] = CP_anon


        if Count_S == 0:
            return None

        # build the domias CI and CP
        domias_n_success = attack_results['domias']['n_success']
        domias_n_attacks = attack_results['domias']['n_attacks']
        domias_y_pred = attack_results['domias']['y_pred']

        C_domias = domias_n_success/domias_n_attacks
        S_for_domias = Sum_S/Count_S
        CI_domias = (C_domias-S_anon)/(1-S_for_domias)
        CP_domias = sum(domias_y_pred)/(len(domias_y_pred)+1)

        defenseGrid1 = [
            (1, 1, 0), (1, .01, .1), (1, .001, .3), (1, .0001, .7), (1, .00001, 1),
            (.95, 1, .1), (.95, .01, .3), (.95, .001, .7), (.95, .0001, .8), (.95, .00001, 1),
            (.90, 1, .3), (.90, .01, .6), (.90, .001, .8), (.90, .0001, .9), (.90, .00001, 1),
            (.75, 1, .7), (.75, .01, .9), (.75, .001, .95), (.75, .0001, 1), (.75, .00001, 1),
            (.50, 1, .95), (.50, .01, .95), (.50, .001, 1), (.50, .0001, 1), (.5, .00001, 1),
            (0, 1, 1), (0, .01, 1), (0, .001, 1), (0, .0001, 1), (0, .00001, 1)
        ]

        def getInterpolatedValue(val0, val1, scoreGrid):
            """Compute interpolated value from grid of mapping tuples

            This routine takes as input a list of tuples ("grid") of the form
            `(val0,val1,score)`. It maps (val0,val1) values to a corresponding
            score. It returns a score that is interpolated between the
            scores in the grid. An example of such a grid can be found in
            gdaScore.py, called `_defenseGrid1`. Note that val0 and val1 must
            go in descending order as shown. Input values that are above the
            highest val0 and val1 values will take the score of the first
            entry. Input values that are below the lowest val0 and val1 will
            take the score of the last entry.
            """
            scoreAbove = -1
            scoreBelow = -1
            for tup in scoreGrid:
                tup0 = tup[0]
                tup1 = tup[1]
                score = tup[2]
                if val0 <= tup0 and val1 <= tup1:
                    tup0Above = tup0
                    tup1Above = tup1
                    scoreAbove = score
            for tup in reversed(scoreGrid):
                tup0 = tup[0]
                tup1 = tup[1]
                score = tup[2]
                if val0 >= tup0 and val1 >= tup1:
                    tup0Below = tup0
                    tup1Below = tup1
                    scoreBelow = score
            if scoreAbove == -1 and scoreBelow == -1:
                return None
            if scoreAbove == -1:
                return scoreBelow
            if scoreBelow == -1:
                return scoreAbove
            if scoreAbove == scoreBelow:
                return scoreAbove
            # Interpolate by treating as right triangle with tup0 as y and
            # tup1 as x
            yLegFull = tup0Above - tup0Below
            xLegFull = tup1Above - tup1Below
            hypoFull = math.sqrt((xLegFull ** 2) + (yLegFull ** 2))
            yLegPart = val0 - tup0Below
            xLegPart = val1 - tup1Below
            hypoPart = math.sqrt((xLegPart ** 2) + (yLegPart ** 2))
            frac = hypoPart / hypoFull
            interpScore = scoreBelow - (frac * (scoreBelow - scoreAbove))
            return interpScore

        defense_score = {}
        defense_score['anon_inference'] = {}
        for col in attack_results['anon_inference']:
            defense_score['anon_inference'][col] = getInterpolatedValue(anon_CICP[col]['ci'], anon_CICP[col]['cp'], defenseGrid1)
        defense_score['domias'] = getInterpolatedValue(CI_domias, CP_domias, defenseGrid1)

        return defense_score

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