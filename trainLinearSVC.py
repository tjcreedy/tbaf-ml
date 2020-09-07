#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Thomas J. Creedy
"""

# Imports

import sys
import argparse
import pickle
import time
import math
import warnings

import matplotlib.colors as mplcol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from collections import defaultdict
from functools import partial
from itertools import product

from numpy.ma import MaskedArray
from scipy.stats import rankdata
from matplotlib.backends.backend_pdf import PdfPages

from sklearn import model_selection, svm, metrics
from sklearn.exceptions import ConvergenceWarning

from sklearn.base import is_classifier, clone
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (_aggregate_score_dicts,
                                                 _fit_and_score)
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.utils.validation import indexable, _check_fit_params

import textwrap as _textwrap

# Global variables

# Class definitions

class MultilineFormatter(argparse.HelpFormatter):
    def _fill_text(self, text, width, indent):
        text = self._whitespace_matcher.sub(' ', text).strip()
        paragraphs = text.split('|n ')
        multiline_text = ''
        for paragraph in paragraphs:
            formatted_paragraph = _textwrap.fill(paragraph, width,
                                                 initial_indent=indent,
                                                 subsequent_indent=indent
                                                 ) + '\n\n'
            multiline_text = multiline_text + formatted_paragraph
        return multiline_text

class Range(argparse.Action):
    def __init__(self, minimum=None, maximum=None, *args, **kwargs):
        self.min = minimum
        self.max = maximum
        kwargs["metavar"] = "[%d-%d]" % (self.min, self.max)
        super(Range, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        if not (self.min <= value <= self.max):
            msg = 'invalid choice: %r (choose from [%d-%d])' % \
                (value, self.min, self.max)
            raise argparse.ArgumentError(self, msg)
        setattr(namespace, self.dest, value)

class SearchParam():
    def __init__(self, values = None, start = None, stop = None, step = None, 
                 resolver = None, unresolver = None):
        self.bestparam = None
        self.bestsource = None
        if values:
            self.source = set(values)
            self.params = set(values)
            self.resolve = lambda x: x
            self.unresolve = self.resolve
            self.step = None
        else:
            try:
                actualstart = min([start, stop])
                actualstop = max([start, stop])
                self.source = set(np.arange(actualstart, actualstop + step,
                                               step, dtype = float))
                self.step = step
            except:
                raise ValueError("start, stop and step, or values required")
        if resolver or unresolver:
            if not unresolver or not resolver:
                raise ValueError("both or neither of resolver and unresolver "
                                 "required")
            self.resolve = resolver
            self.unresolve = unresolver
            self.params = set([self.resolve(v) for v in self.source])
        self.tested = set()
    
    def get_untested_params(self):
        out = {p for p in self.params if p not in self.tested}
        self.tested.update(out)
        return(sorted(list(out)))
    
    def get_minmax_untested_params(self):
        out = {p for p in self.params if p not in self.tested}
        out = {min(out), max(out)}
        self.tested.update(out)
        return(sorted(list(out)))
    
    def any_untested_params(self):
        unt = [p for p in self.params if p not in self.tested]
        return(len(unt) > 0)
    
    def add_values(self, values):
        self.source.update(values)
        self.params.update([self.resolve(v) for v in values])
    
    def set_best_param(self, best):
        self.bestparam = best
        self.bestsource = self.unresolve(best)
        self.tested.remove(best)
    
    def set_best_make_new(self, best):
        self.set_best_param(best)
        if self.step is None:
            return
        if min(self.source) < self.bestsource < max(self.source):
            self.step = 0.5 * self.step
            newsource = {self.bestsource + m * self.step for m in [-1, 1]}
            
        else:
            mult = [1, 2]
            if self.bestsource == min(self.source):
                mult = [-1 * m for m in mult]
            newsource = {self.bestsource + m * self.step for m in mult}
        self.source.update(newsource)
        self.params.update(self.resolve(s) for s in newsource)


class SearchParamSet():
    def __init__(self, params):
        self.paramset = dict()
        for k, v in params.items():
            if (type(v) is tuple and len(v) == 5 
                and callable(v[3]) and callable(v[4])):
                kwargs = {k: a for k, a in 
                          zip(['start', 'stop', 'step', 
                               'resolver', 'unresolver'], v)}
            else:
                kwargs = {'values': v}
            
            self.paramset[k] = SearchParam(**kwargs)
            
    
    def get_minmax_untested_params(self):
        out = dict()
        for k, v in self.paramset.items():
            out[k] = v.get_minmax_untested_params()
        return(out)
    
    def get_untested_params(self):
        out = dict()
        for k, v in self.paramset.items():
            out[k] = v.get_untested_params()
        return(out)
    
    def set_best_make_new(self, best):
        for k, v in best.items():
            self.paramset[k].set_best_make_new(v)
    
    def get_all_params(self):
        out = dict()
        for k, v in self.paramset.items():
            out[k] = sorted(list(v.params))
        return(out)
    
    def any_untested_params(self):
        return(any(v.any_untested_params() for v in self.paramset.values()))

class GridSearchCV_custom(model_selection.GridSearchCV):
    
    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Run fit with all sets of parameters.
        
        Parameters
        ----------
        
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        
        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        # self, X, y, groups, fit_params = gs, train, cls, None, {}
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))
        
        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)
        
        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, str) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers) and not callable(self.refit):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key or a "
                                 "callable to refit an estimator with the "
                                 "best parameter setting on the whole "
                                 "data and make the best_* attributes "
                                 "available for that metric. If this is "
                                 "not needed, refit should be set to "
                                 "False explicitly. %r was passed."
                                 % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'
        
        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)
        
        n_splits = cv.get_n_splits(X, y, groups)
        
        base_estimator = clone(self.estimator)
        
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch=self.pre_dispatch)
        
        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    return_n_iter=True,
                                    return_cm=True,
                                    return_roc=True,
                                    return_prc=True,
                                    return_threshc=True,
                                    return_estimator=False,
                                    error_score=self.error_score,
                                    verbose=self.verbose)
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []

            def evaluate_candidates(candidate_params):
                # candidate_params = model_selection.ParameterGrid(self.param_grid)
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                              n_splits, n_candidates, n_candidates * n_splits))

                out = parallel(delayed(_fitnscore_cust)(clone(base_estimator),
                                                       X, y,
                                                       train=train, test=test,
                                                       parameters=parameters,
                                                       **fit_and_score_kwargs)
                               for parameters, (train, test)
                               in product(candidate_params,
                                          cv.split(X, y, groups)))

                if len(out) < 1:
                    raise ValueError('No fits were performed. '
                                     'Was the CV iterator empty? '
                                     'Were there no candidates?')
                elif len(out) != n_candidates * n_splits:
                    raise ValueError('cv.split and cv.get_n_splits returned '
                                     'inconsistent results. Expected {} '
                                     'splits, got {}'
                                     .format(n_splits,
                                             len(out) // n_candidates))

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                nonlocal results
                results = self._format_results(
                       all_candidate_params, scorers, n_splits, all_out)
                return results

            self._run_search(evaluate_candidates)

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                   self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                self.best_index_ = results["rank_test_%s"
                                           % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                                           self.best_index_]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(clone(base_estimator).set_params(
                **self.best_params_))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self

    def _format_results(self, candidate_params, scorers, n_splits, out):
        # candidate_params, scorers, n_splits, out = all_candidate_params, scorers, n_splits, all_out
        n_candidates = len(candidate_params)
        
        values = dict()
        for d in out:
            for k, v in d.items():
                if k in values:
                    values[k].append(v)
                else:
                    values[k] = [v]
        
        # test_score_dicts, train_score dicts and confmat_dicts are lists of
        # dictionaries and we make them into dict of lists
        test_scores = _aggregate_score_dicts(values['test_scores'])
        if 'train_scores' in values:
            train_scores = _aggregate_score_dicts(values['train_scores'])
        if 'confusion_matrix' in values:
            confmats = _aggregate_score_dicts(values['confusion_matrix'])
        
        results = {}
        
        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]
            
            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds
            
            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)
        
        for s in ['fit_time', 'score_time', 'n_iter']:
            # s = 'n_iter'
           if s in values:
               _store(s, values[s])
        
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value
        
        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params
        
        # NOTE test_sample counts (weights) remain the same for all candidates
        if 'n_test_samples' in values:
            test_sample_counts = np.array(values['n_test_samples'][:n_splits],
                                          dtype=np.int)
        
        if self.iid != 'deprecated':
            warnings.warn(
                "The parameter 'iid' is deprecated in 0.22 and will be "
                "removed in 0.24.", FutureWarning
            )
            iid = self.iid
        else:
            iid = False
        
        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights=test_sample_counts if iid else None)
        if self.return_train_score:
            _store('train_%s' % scorer_name, train_scores[scorer_name],
                   splits=True)
        if 'confusion_matrix' in values:
            for bin_name, bin_values in confmats.items():
                _store(bin_name, bin_values)
            
        
        # Store the plotting dicts
        for n in ['roc_values', 'prc_values', 'threshc_values']:
            if n in values:
                results[n] = np.array(values[n]).reshape(n_candidates, 
                                                         n_splits)
        
        return results
    
    def post_refit(self, X, y, refit_metric, **fit_params):
        # X, y, refit_metric, self = train, cls, rs, search
        self.refit = refit_metric
        results = self.cv_results_
        base_estimator = self.estimator
        
        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            # If callable, refit is expected to return the index of the best
            # parameter set.
            if callable(self.refit):
                self.best_index_ = self.refit(results)
                if not isinstance(self.best_index_, numbers.Integral):
                    raise TypeError('best_index_ returned is not an integer')
                if (self.best_index_ < 0 or
                   self.best_index_ >= len(results["params"])):
                    raise IndexError('best_index_ index out of range')
            else:
                results["rank_test_%s" % refit_metric] = np.asarray(
                            rankdata(-results["mean_test_%s" % refit_metric],
                                     method='min'), dtype=np.int32)
                self.best_index_ = results["rank_test_%s"
                                           % refit_metric].argmin()
                self.best_score_ = results["mean_test_%s" % refit_metric][
                                           self.best_index_]
            self.best_params_ = results["params"][self.best_index_]
    
        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(clone(base_estimator).set_params(
                **self.best_params_))
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

# Function definitions


def _fitnscore_cust(estimator, X, y, scorer, train, test, verbose,
                   parameters, fit_params, return_train_score=False,
                   return_parameters=False, return_n_test_samples=False,
                   return_times=False, return_n_iter=False, return_cm=False,
                   return_roc=False, return_prc=False, return_threshc=False,
                   return_estimator=False,
                   error_score=np.nan):
    # estimator, scorer, return_train_score, return_parameters, return_n_test_samples, return_times, return_n_iter, return_cm, return_roc, return_prc, return_threshc, return_estimator, error_score, verbose = clone(base_estimator), scorers, True, True, True, True, True, True, True, True, True, True, np.nan, 5
    # parameters, (train, test) = next(product(candidate_params, cv.split(X, y, groups)))
    
    
    import numbers
    from traceback import format_exc
    from sklearn.utils.validation import _check_fit_params, _num_samples
    from sklearn.utils.metaestimators import _safe_split
    from sklearn.exceptions import FitFailedWarning
    from sklearn.model_selection._validation import _score
    from sklearn.utils import _message_with_time
    
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                                    for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    train_scores = {}
    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn("Estimator fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exc()),
                          FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, scorer)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(estimator, X_train, y_train, scorer)
        n_iter = estimator.n_iter_
        if return_cm:
            cm = dict(zip(['true_negatives', 'false_positives',
                           'false_negatives', 'true_positives'],
                          [0] * 4))
            for true, pred in zip(y_test, estimator.predict(X_test)):
                # true, pred = next(zip(y_test, estimator.predict(X_test)))
                s = sum([true, pred])
                if s == 2:
                    cm['true_positives'] += 1
                elif s == 0:
                    cm['true_negatives'] += 1
                elif true == 1:
                    cm['false_negatives'] += 1
                elif pred == 1:
                    cm['false_positives'] += 1
                else:
                    raise Exception(f"Values {true} and {pred} not valid for"
                                    " computing confusion matrix")
        if return_roc or return_prc or return_threshc:
            df = estimator.decision_function(X_test)
            if return_roc:
                roc = metrics.roc_curve(y_test, df)
                auc = metrics.auc(roc[1], roc[2])
            if return_threshc or return_prc:
                prc = metrics.precision_recall_curve(y_test, df)
                ap = metrics.average_precision_score(y_test, df)
            if return_roc:
                roc_out = {'fpr': roc[0], 'tpr': roc[1], 'auc': auc}
            if return_prc:
                prc_out = {'rcl': prc[1], 'pre': prc[0], 'ap': ap}
            if return_threshc:
                threshc_out = {'thr': prc[2], 'rcl': prc[1][:-1], 
                               'pre': prc[0][:-1]}
    
    if verbose > 2:
        if isinstance(test_scores, dict):
            for scorer_name in sorted(test_scores):
                msg += ", %s=" % scorer_name
                if return_train_score:
                    msg += "(train=%.3f," % train_scores[scorer_name]
                    msg += " test=%.3f)" % test_scores[scorer_name]
                else:
                    msg += "%.3f" % test_scores[scorer_name]
        else:
            msg += ", score="
            msg += ("%.3f" % test_scores if not return_train_score else
                    "(train=%.3f, test=%.3f)" % (train_scores, test_scores))
    
    if verbose > 1:
        total_time = score_time + fit_time
        print(_message_with_time('CV', msg, total_time))
    
    ret = {'test_scores': test_scores}
    if return_train_score:
        ret['train_scores'] = train_scores
    if return_n_test_samples:
        ret['n_test_samples'] = _num_samples(X_test)
    if return_times:
        ret['fit_time'] = fit_time
        ret['score_time'] = score_time
    if return_parameters:
        ret['parameters'] = parameters
    if return_n_iter:
        ret['n_iter'] = n_iter
    if return_estimator:
        ret['estimator'] = estimator
    if return_cm:
        ret['confusion_matrix'] = cm
    if return_roc:
        ret['roc_values'] = roc_out
    if return_prc:
        ret['prc_values'] = prc_out
    if return_threshc:
        ret['threshc_values'] = threshc_out
    return ret


def listmult(lis):
    result = 1
    for i in lis:
        result = result * i
    return(result)

def analyse_results(search, scorers, output, train, cls, strat):
    # search, output = grid_search, args.output
    
    # Curve names
    curves = ['roc_values', 'prc_values', 'threshc_values']
    
    # Set up outputs
    models = dict()
    models['fullsearch'] = search
    
    # Open a pdf to write to
    pdf = PdfPages(f"{output}.pdf")
    
    # Split training data up into test and train data
    ttsplit = model_selection.train_test_split(train, cls, stratify = strat)
    data_train, data_test, cls_train, cls_test  = ttsplit
    
    print("Split training data as follows to score hyperparameter search:")
    print(pd.DataFrame([countcls(cls_train, [0,1]), 
                        countcls(cls_test, [0,1])],
                        columns = ['invalid', 'valid'],
                        index = ['train', 'test']))
    
    
    
    # Work through the scorers refitting and generating plotting data
    roc = {'fpr': dict(), 'tpr': dict(), 'thr': dict(), 'auc': dict()}
    prc = {'pre': dict(), 'rcl': dict(), 'thr': dict(), 'ap': dict()}
    cms = []
    for rs in scorers.keys():
        #rs = list(scorers.keys())[0]
        
        # Refit with all training data for this score to get individual
        # plots
        print(f"\nRefitting Grid Search result with {rs}")
        search.post_refit(train, cls, rs)
        bp = search.best_params_
        i = np.where(search.cv_results_['params'] == bp)[0][0]
            #Curves
        for c in curves:
            # c = 'prc_values'
            pdf.savefig(plot_kfold_curves(search.cv_results_[c][i], 
                                          search.n_splits_,
                                          rs))
            #Confusion matrices
        cm_values = []
        classes = ['negatives', 'positives']
        bools = ['true', 'false']
        for d in [1, -1]:
            r = []
            for b, c in zip(bools[::d], classes):
                mean = np.around(search.cv_results_[f"mean_{b}_{c}"][i], 2)
                std = np.around(search.cv_results_[f"std_{b}_{c}"][i], 2)
                r.append(f"{mean} ± {std} SD")
            cm_values.append(r)
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.table(cellText = cm_values,
                 rowLabels = ["true_0", "true_1"],
                 colLabels = ["predicted_0", "predicted_1"],
                 loc = 'center')
        ax.set_title( "Mean and Standard Deviation Confusion Matrix\n"
                     f"values for the {search.n_splits_} test-train split\n"
                     f"model fittings with the best {rs}")
        pdf.savefig(fig)
        
        # Save final model
        models['rs'] = clone(search.best_estimator_)
        
        # Do fit with train/test split
        be = clone(search.best_estimator_)
        be.fit(data_train, cls_train)
        df = be.decision_function(data_test)
        
        # Compute Receiver Operating Characteristic curve values
        out = metrics.roc_curve(cls_test, df)
        roc['fpr'][rs], roc['tpr'][rs], roc['thr'][rs] = out
        roc['auc'][rs] = metrics.auc(roc['fpr'][rs], roc['tpr'][rs])
        
        # Compute Precision Recall curve values
        out = metrics.precision_recall_curve(cls_test, df)
        prc['pre'][rs], prc['rcl'][rs], prc['thr'][rs] = out
        prc['ap'][rs] = metrics.average_precision_score(cls_test, df)
        
        # Plot confusion matrix
        cmplt = metrics.plot_confusion_matrix(be, data_test, cls_test)
        cmplt.ax_.set_title(f"Confusion matrix for fitting optimised for {rs}")
        cms.append(cmplt.figure_)
        
    
    # Do plotting
    colours = dict(zip(scorers.keys(), 'bgr'))
    
    # ROC curve
    fig, ax = plt.subplots()
    for rs in scorers.keys():
        ax.plot(roc['fpr'][rs], roc['tpr'][rs], c = colours[rs],
                 label = (f"Optimised for {rs} (area = "
                          f"{np.around(roc['auc'][rs], 2)})"))
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic Curves optimised for '
              'different scores')
    ax.legend(loc = 'lower right')
    pdf.savefig(fig)
    
    # PR curve
    fig, ax = plt.subplots()
    for rs in scorers.keys():
        ax.plot(prc['rcl'][rs], prc['pre'][rs], c = colours[rs],
                 label = (f"Optimised for {rs} "
                          f"(AP = {np.around(prc['ap'][rs], 2)})"))
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves optimised for '
              'different scores')
    ax.legend(loc = 'lower right')
    pdf.savefig(fig)
    
    # Thresholds curve
    fig, ax = plt.subplots()
    for rs in scorers.keys():
        #rs = list(scorers.keys())[0]
        ax.plot(prc['thr'][rs], prc['pre'][rs][:-1], '--', c = colours[rs],
                 label = f"Precision scores by threshold for {rs}")
        ax.plot(prc['thr'][rs], prc['rcl'][rs][:-1], c = colours[rs],
                 label = f"Recall scores by threshold for {rs}")
    ax.set_ylim([0.0, 1.05])
    ax.set_xscale('symlog')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision')
    ax.set_title('Values of precision and recall scores over varying decision '
              'thresholds')
    ax.legend(loc = 'lower right')
    pdf.savefig(fig)
    
    # Confusion matrices
    for cm in cms:
        pdf.savefig(cm)
    
    pdf.close()
    
    results_out = {k:v for k,v in search.cv_results_.items() 
                           if k not in curves}
    pd.DataFrame(results_out).to_csv(f"{output}_gscvresults.csv")
    
    return(models)

def plot_kfold_curves(dicts, n, score):
    # dicts, n, score = search.cv_results_[c][i], search.n_splits_, rs
    dataheads = set(dicts[1].keys())
    title = (f" curves\nfor the {n} test-train split model fittings with\nthe "
             f"best {score}")
    if dataheads == {'auc', 'fpr', 'tpr'}:
        x, y1, y2, v, f = 'fpr', 'tpr', None, 'auc', 1
        xlab, ylab, title = ('False Positive Rate',
                             'True Positive Rate',
                             (f"Receiver Operating Characteristic{title}, "
                              "AUC = "))
    elif dataheads == {'rcl', 'pre', 'ap'}:
        x, y1, y2, v, f = 'rcl', 'pre', None, 'ap', None
        xlab, ylab, title = ('Recall',
                             'Precision',
                             f"Precision-Recall{title}, average precision = ")
    elif dataheads == {'thr', 'rcl', 'pre'}:
        x, y1, y2, v, f = 'thr', 'rcl', 'pre', None, None
        xlab, ylab, title = ('Threshold',
                             'Score',
                             f"Precision (- -) and Recall (---) value{title}")
    
    if n <= 10:
        colours = list(mplcol.TABLEAU_COLORS)[:n]
    else:
        colours = ['r'] * n
    
    fig, ax = plt.subplots()
    sumvals = []
    mean_x = np.linspace(0, 1, 100)
    interp_y1 = []
    
    for d, col in zip(dicts, colours):
        # d, col = list(zip(dicts, colours))[0]
        a = 0.8 if y2 else 0.3
        ax.plot(d[x], d[y1], c = col, alpha = a, lw = 1)
        dxsort = sorted(range(len(d[x])), key = lambda k: d[x][k])
        interp_y1.append([0] + np.interp(mean_x, d[x][dxsort], d[y1][dxsort]))
        if y2:
            ax.plot(d[x], d[y2], '--', c = col)
        if v:
            sumvals.append(d[v])
    
    if y2:
        ax.set_xscale('symlog')
    else: 
        mean_y1 = np.mean(interp_y1, axis = 0)
        if f: mean_y1[-1] = f
        ax.plot(mean_x, mean_y1, color = 'black', lw = 2, alpha = 0.8,
                label = "mean")
        std_y1 = np.std(interp_y1, axis = 0)
        y1_upper = np.minimum(mean_y1 + std_y1, 1)
        y1_lower = np.maximum(mean_y1 - std_y1, 0)
        ax.fill_between(mean_x, y1_lower, y1_upper, color = 'grey',
                        alpha = 0.3, label = '± 1 std. dev.')
        ax.legend(loc = "lower right")
    
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    if v is not None:
        mean = np.around(np.mean(sumvals), 2)
        std = np.around(np.std(sumvals), 2)
        title = f"{title}{mean} ±{std} SD"
    
    ax.set_title(title)
    return(fig)


def countcls(cls, v):
    if type(v) is int:
        return(len([c for c in cls if c == v]))
    else:
        return([len([c for c in cls if c == vi]) for vi in v])

def param_count(params):
    lns = [len(v) for v in params.values()]
    return(listmult(lns))

def iterativeGridSearchCV(clf, paramspecs, scorers, data, cls,
                          gscvkwargs, args, threshold = 1e-6, maxiters = 10):
    # data, cls, threshold, maxiters = data_train, cls_train, 1e-8, 10 
    
    if not 0 < threshold < 1 :
        raise ValueError("threshold for score change should be < 1 and > 0")
    
    param_search = SearchParamSet(paramspecs)
    
    gs = None
    s = 0
    bestparams = None
    results = None
    for scorer in list(scorers.keys()):
        #scorer = list(scorers.keys())[2]
        s += 1
        score = 0
        
        if gs is not None:
            gs.post_refit(data, cls, scorer)
            score = gs.best_score_
            bestparams = gs.best_params_
        
        scorechange = 1
        n = 0
        while scorechange > threshold and n < maxiters:
            n += 1
            # Check for untested params, if none use current best params 
            # to derive some
            if not param_search.any_untested_params():
                param_search.set_best_make_new(bestparams)
            params = param_search.get_untested_params()
            
            print(f"Running Grid Search on scorer {s} of {len(scorers)}, "
                  f"iteration {n}. Testing {param_count(params)} combinations "
                  f"of {len(params)} hyperparameters, current score {score}")
            
            # Define and fit the model
            
            gs = GridSearchCV_custom(clf, params,
                                     refit = scorer,
                                     **gscvkwargs)
            
            with warnings.catch_warnings(record = True):
                warnings.filterwarnings('ignore', 
                                        category = ConvergenceWarning)
                gs.fit(data, cls)
            
            bestparams = gs.best_params_
            
            # Extract the results and compile
            if results is None:
                results = gs.cv_results_
            else:
                nres = gs.cv_results_
                isnew = [not any([np == rp for rp in results['params']]) 
                            for np in nres['params']]
                for k, vs in nres.items():
                    #k, vs = 'roc_values', nres['roc_values']
                    results[k] = np.append(results[k],
                                          [v for v, i in zip(vs, isnew) if i], 
                                          axis = 0)
                gs.cv_results_ = results
            
            # Extract the score and compute change
            scorechange = gs.best_score_ - score
            score = gs.best_score_
    
    return(gs)

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax

def getcliargs(arglist = None):
    
    parser = argparse.ArgumentParser(description="""
        description:
        |n
        Text
        |n
        Text
        """,formatter_class=MultilineFormatter)
    
    parser._optionals.title = "arguments"
    parser.add_argument('-d', '--data',
                        help = 'path to training data',
                        type = str)
    parser.add_argument('-t', '--threads',
                        help = 'number of cores',
                        type = int, default = 1)
    parser.add_argument('-m', '--maxiter',
                        help = 'the maximum number of iterations for the '
                               'estimator',
                        type = int, default = 1000)
    parser.add_argument('-o', '--output',
                        help = 'prefix path to output results',
                        type = str)
    
    args = parser.parse_args(arglist) if arglist else parser.parse_args()
    
    # Checking
    #parser.error
    
    sys.stderr.flush()
    return(args)

# Main

def main():
    
    args = getcliargs()
    #args = getcliargs(['-d','testdata/amm/prepped.pickle','-t','-1','-o','SVCout'])
    # args = getcliargs(['-d','testdata/cccpmeta_rand1000/prepped_trainingdata.csv','-t','-1','-o','SVCout'])
    train = pd.read_csv(args.data, index_col = 0)
    cls = train.pop('class')
    strat = train.pop('stratum')
    # Drop this data because n_stops and n_nucs were used to identify the 
    # known invalid data
    train = train.drop(['n_stops', 'n_nucs'], axis = 1)
    
    print(f"\nLoaded {len(cls)} total training data points")
    
    scorers = {
        'precision_score': metrics.make_scorer(metrics.precision_score, 
                                               zero_division = 0),
        'recall_score': metrics.make_scorer(metrics.recall_score),
        'accuracy_score': metrics.make_scorer(metrics.accuracy_score)
    }
    
    gscvkwargs = {'cv': model_selection.StratifiedShuffleSplit(n_splits = 10), 
                  'return_train_score': False,
                  'n_jobs': args.threads,
                  'pre_dispatch': '2*n_jobs',
                  'scoring': scorers,
                  'verbose': 1}
    
    paramspecs = {'C': (-10, 4, 1, 
                         lambda x : 10 ** x, 
                         lambda x : math.log10(x)),
                  'tol': (-8, 2, 1, 
                          lambda x : 10 ** x, 
                          lambda x : math.log10(x))
                  }
    
    clf = svm.LinearSVC(dual = False, max_iter = args.maxiter)
    
#    ps = SearchParamSet(paramspecs)
#    pg = ps.get_minmax_untested_params()
#    gs = GridSearchCV_custom(clf, pg, refit = 'precision_score', **gscvkwargs)
#    gs.fit(train, cls)
    gs.post_refit(train, cls, 'accuracy_score')
    grid_search = iterativeGridSearchCV(clf, paramspecs, scorers, train,
                                        cls,  gscvkwargs, args, 1e-8, 10)
    
    print("\nCompleted Grid Search")
    
    models = analyse_results(grid_search, scorers, args.output, train, cls,
                             strat)
    
    with open(f"{args.output}_bestestimators.pickle", 'wb') as oh:
        pickle.dump(models, oh)


if __name__ == "__main__":
    main()
    exit()


#    CODE TO AUTOMATICALLY STEP UP MAX_ITERS UNTIL NO WARNINGS
#    params = {'C': [1e-08, 1000.0], 
#              'tol': [1e-10, 100.0]}
#    iterfail = True
#    iterrun = 0
#    iterstart = 1000
#    iterstep = 1000
#    while iterfail and iterrun < 10:
#        
#        curriter = iterstart + iterrun * iterstep
#        print(f"Run {iterrun} with max_iter {curriter}...",)
#        
#        iterrun += 1
#        
#        with warnings.catch_warnings():
#            warnings.filterwarnings('error')
#            try:
#                params['max_iter'] = [curriter]
#                gs = model_selection.GridSearchCV(clf,
#                                          params,
#                                          refit = 'accuracy_score',
#                                          **gscvkwargs)
#                gs.fit(data_train, cls_train)
#                print("fitting completed")
#                iterfail = False
#                
#            except ConvergenceWarning:
#                print("convergence warning caught")
#                iterfail = True
