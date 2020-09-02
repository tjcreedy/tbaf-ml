#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Thomas J. Creedy
"""

# Imports

import sys
import argparse
import pandas
import pickle
import numpy
import time
import math
import warnings
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.backends.backend_pdf import PdfPages
from itertools import cycle
from sklearn import model_selection, svm, metrics
from sklearn.utils.fixes import loguniform
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
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
                self.source = set(numpy.arange(actualstart, actualstop + step,
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



# Function definitions

def post_refit(self, X, y, refit_metric, **fit_params):
    
    
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
    



def listmult(lis):
    result = 1
    for i in lis:
        result = result * i
    return(result)

def analyse_results(search, scorers, output, new, train, cls, data_train, 
                    cls_train, data_test, cls_test):
    
    # Set up outputs
    pred = pandas.DataFrame(index = new.index)
    retain = dict()
    
    # Open a pdf to write to
    pdf = PdfPages(f"{output}.pdf")
    
    # Work through the scorers refitting and generating plotting data
    roc = {'fpr': dict(), 'tpr': dict(), 'thr': dict(), 'auc': dict()}
    prc = {'pre': dict(), 'rcl': dict(), 'thr': dict(), 'ap': dict()}
    for rs in scorers.keys():
        #rs = list(scorers.keys())[0]
        
        # Refit for this score
        print(f"\nRefitting Grid Search result with {rs}")
        post_refit(search, data_train, cls_train, rs)
        df = search.decision_function(data_test)
        be = search.best_estimator_
        
        # Do final fit with this estimator
        best_clf = clone(be)
        best_clf.fit(train, cls)
        pred[f"prediction_{rs}"] = best_clf.predict(new)
        pred[f"decision_{rs}"] = best_clf.decision_function(new)
        retain = pred.index[pred[f"prediction_{rs}"] == 1].tolist()
        
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
        pdf.savefig(cmplt.figure_)
    
    # Do plotting
    colours = dict(zip(scorers.keys(), 'bgr'))
    
    # ROC curve
    
    plt.figure()
    for rs in scorers.keys():
        plt.plot(roc['fpr'][rs], roc['tpr'][rs], c = colours[rs],
                 label = f"ROC curve for {rs} (area = {roc['auc'][rs]})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves optimised for '
              'different scores')
    plt.legend(loc = 'lower right')
    pdf.savefig(plt)
    
    # PR curve
    plt.figure()
    for rs in scorers.keys():
        plt.plot(prc['rcl'][rs], prc['pre'][rs], c = colours[rs],
                 label = f"PR curve for {rs} (AP = {prc['ap'][rs]})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves optimised for '
              'different scores')
    plt.legend(loc = 'lower right')
    pdf.savefig(plt)
    
    # Thresholds curve
    plt.figure()
    for rs in scorers.keys():
        #rs = list(scorers.keys())[0]
        plt.plot(prc['thr'][rs], prc['pre'][rs][:-1], '--', c = colours[rs],
                 label = f"Precision scores by threshold for {rs}")
        plt.plot(prc['thr'][rs], prc['rcl'][rs][:-1], c = colours[rs],
                 label = f"Recall scores by threshold for {rs}")
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.title('Values of precision and recall scores over varying  decision '
              'thresholds')
    plt.legend(loc = 'lower right')
    pdf.savefig(plt)
    
    pdf.close()
    
    pandas.DataFrame(search.cv_results_).to_csv(f"{output}_gscvresults.csv")
    pred.to_csv(f"{output}_predictions.csv")
    
    return(retain)

def countcls(cls, v):
    if type(v) is int:
        return(len([c for c in cls if c == v]))
    else:
        return([len([c for c in cls if c == vi]) for vi in v])

def param_count(params):
    lns = [len(v) for v in params.values()]
    return(listmult(lns))

def iterativeGridSearchCV(clf, paramspecs, scorers, data_train, cls_train, 
                          gscvkwargs, args, threshold = 1e-8, maxiters = 10):
    if not 0 < threshold < 1 :
        raise ValueError("threshold for score change should be < 1 and > 0")
    
    param_search = SearchParamSet(paramspecs)
    
    gs = None
    s = 0
    bestparams = None
    results = None
    for scorer in list(scorers.keys()):
        #scorer = list(scorers.keys())[0]
        s += 1
        score = 0
        
        if gs is not None:
            post_refit(gs, data_train, cls_train, scorer)
            score = gs.best_score_
            bestparams = gs.best_params_
        
        scorechange = 1
        n = -1 if s == 1 else 0
        while scorechange > threshold and n < maxiters:
            n += 1
            # Check for untested params, if none use current best params 
            # to derive some
            if not param_search.any_untested_params():
                param_search.set_best_make_new(bestparams)
            # Retrieve untested params
            if n == 0:
                # If this is the very first run, just get the maximum values
                # to test that the number of iterations will be sufficient
                params = param_search.get_minmax_untested_params()
                n += 1
            else:
                params = param_search.get_untested_params()
            
            print(f"Running Grid Search on scorer {s} of {len(scorers)}, "
                  f"iteration {n}. Testing {param_count(params)} combinations "
                  f"of {len(params)} hyperparameters, current score {score}")
            
            # Define and fit the model
            gs = model_selection.GridSearchCV(clf,
                                              params,
                                              refit = scorer,
                                              **gscvkwargs)
            
            with warnings.catch_warnings():
                if args.allownonconvergence:
                    warnings.filterwarnings('ignore', 
                                            category = ConvergenceWarning)
                    gs.fit(data_train, cls_train)
                else:
                    warnings.filterwarnings('error')
                    try:
                        gs.fit(data_train, cls_train)
                    except ConvergenceWarning:
                        exit("Error: one or more fits failed to converge. "
                             "Increase -m/--maxiter or use "
                             "--allownonconvergence")
            
            gs.fit(data_train, cls_train)
            
            bestparams = gs.best_params_
            
            # Extract the results and compile
            if results is None:
                results = pandas.DataFrame(gs.cv_results_)
            else:
                nres = pandas.DataFrame(gs.cv_results_)
                isnew = [not any([np == rp for rp in results['params']]) 
                            for np in nres['params']]
                nres = nres.loc[isnew]
                results = pandas.concat([results, nres], axis = 0)
                gs.cv_results_ = dict(zip(results.columns, 
                                          results.T.values))
            
            # Extract the score and compute change
            scorechange = gs.best_score_ - score
            score = gs.best_score_
    
    return(gs)

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
                        help = 'path to prepared data',
                        type = str)
    parser.add_argument('-t', '--threads',
                        help = 'number of cores',
                        type = int, default = 1)
    parser.add_argument('-m', '--maxiter',
                        help = 'the maximum number of iterations for the '
                               'estimator',
                        type = int, default = 1000)
    parser.add_argument('--allownonconvergence',
                        help = 'don\'t fail if an estimator does not converge',
                        action = 'store_true')
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
    #args = getcliargs(['-d','testdata/cccpmeta_rand2000/prepped.pickle','-t','-1','-o','SVCout'])
    with open(args.data, 'rb') as ih:
        train, cls, strat, new = pickle.load(ih)
    
    # Split training data up into test and train data
    ttsplit = model_selection.train_test_split(train, cls, stratify = strat)
    data_train, data_test, cls_train, cls_test  = ttsplit
    
    print(f"\nLoaded {len(cls)} total training data points, split as follows "
           "for training hyperparameters:")
    print(pandas.DataFrame([countcls(cls_train, [0,1]), 
                            countcls(cls_test, [0,1])],
                           columns = ['invalid', 'valid'],
                           index = ['train', 'test']))
    
    scorers = {
        'precision_score': metrics.make_scorer(metrics.precision_score, 
                                               zero_division = 0),
        'recall_score': metrics.make_scorer(metrics.recall_score),
        'accuracy_score': metrics.make_scorer(metrics.accuracy_score)
    }
    
    gscvkwargs = {'cv': model_selection.StratifiedKFold(n_splits = 10), 
                  'return_train_score': True,
                  'n_jobs': args.threads,
                  'pre_dispatch': '2*n_jobs',
                  'scoring': scorers}
    
    paramspecs = {'C': (-8, 3, 1, 
                         lambda x : 10 ** x, 
                         lambda x : math.log10(x)),
                  'tol': (-10, 2, 1, 
                          lambda x : 10 ** x, 
                          lambda x : math.log10(x))
                  }
    
    clf = svm.LinearSVC(dual = False, max_iter = args.maxiter)
    
    grid_search = iterativeGridSearchCV(clf, paramspecs, scorers, data_train,
                                        cls_train, gscvkwargs, args, 1e-12, 10)
    
    print("\nCompleted Grid Search")
    
    predictions, retentions = analyse_results(grid_search, scorers,
                                              args.output, new, train, cls,
                                              data_train, cls_train, data_test,
                                              cls_test)
    


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
