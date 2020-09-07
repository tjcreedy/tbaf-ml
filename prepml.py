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
from sklearn import impute, preprocessing, model_selection

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

# Function definitions


def parse_scalevalues(path, typ):
    #path, typ = args.input, 'input'
    out = [None] * 2
    indf = pandas.read_csv(path, index_col='name')
    indf = indf.drop(labels=['n_stops', 'n_nucs'], axis=1)
    if typ in 'vi':
        out[0 if typ == 'v' else 1] = indf
    else:
        out = indf
#    else:
#        newi = [i for i, (s, l) in enumerate(zip(indf.n_stops, indf.n_nucs))
#                 if s == 0 and l == 418]
#        indf = indf.drop(labels=['n_stops', 'n_nucs'], axis=1)
#        invalidi = [i for i in range(len(indf.index)) if i not in newi]
#        out[1] = indf.iloc[newi]
#        out[2] = indf.iloc[invalidi]
    return(out)

def process_scaledata(args):
    # Load new data
    new = pandas.read_csv(args.scales, index_col='name')
    # Load training data if present
    train = list()
    for p in [args.validscales, args.invalidscales]:
        if p is not None:
            train.append(pandas.read_csv(p, index_col='name'))
        else:
            train.append(pandas.DataFrame(columns=new.columns))
    lens = [len(n.index) for n in train]
    # Subsample training dataframes if necessary
    if args.trainsize and sum(lens) > args.trainsize:
        if lens[0] < 0.5 * args.trainsize:
            lens[1] = args.trainsize - lens[0]
        elif lens[1] < 0.5 * args.trainsize:
            lens[0] = args.trainsize - lens[1]
        else:
            lens[0], lens[1] = [int(round(0.5 * args.trainsize, 0))] * 2
        for i in range(2):
            if lens[i] != len(train[i].index):
                train[i] = train[i].sample(lens[i])
    # Return concatenated training dataset, list of training classes,
    #   new dataset
    return(pandas.concat(train, axis=0, join='inner'),
           [1] * lens[0] + [0] * lens[1],
           new)
    
def process_abundancedata(path, new):
    #path = args.abundance
    indf = pandas.read_csv(path, sep = '\t', index_col = 0)
    # Standardise by sample totals
    indf = indf.div(indf.sum(axis = 0), axis = 1)
    # Discard sample-wise values and instead sort by magnitude
    a = indf.values
    a.sort(axis = 1)
    indf = pandas.DataFrame(a[:, ::-1], indf.index)
    # Drop any columns with zero totals
    indf = indf.loc[:,indf.sum(axis = 0) > 0]
    # Merge with scales data and return
    return(pandas.concat([new, indf], axis = 1))

def process_known(args, new):
    # Set up known dict
    known = {'v': [], 'i': []}
    # Read in known lists if available
    for t, p in zip('vi', [args.knownvalid, args.knowninvalid]):
        if p:
            known[t] = [x.strip() for x in open(p).readlines()]
    # Check for stops
    known['i'].extend(new.index[new['n_stops'] > 0].tolist())
    # Find unique
    known['i'] = set(known['i'])
    # Remove any invalid from valid
    known['v'] = set(known['v']) - known['i']
    # Generate class list
    tcls = [1 if i in known['v'] else 0 if i in known['i'] else None 
           for i in new.index]
    # Separate new and train
    train = new[[c is not None for c in tcls]]
    new = new[[c is None for c in tcls]]
    tcls = [c for c in tcls if c is not None]
    # Return
    return(train, tcls, new)

def countcls(cls, v):
    if type(v) is int:
        return(len([c for c in cls if c == v]))
    else:
        return([len([c for c in cls if c == vi]) for vi in v])


def getcliargs(arglist = None):
    
    parser = argparse.ArgumentParser(description="""
        description:
        |n
        Text
        |n
        Text
        """,formatter_class=MultilineFormatter)
    
    parser._optionals.title = "arguments"
    parser.add_argument('-s', '--scales',
                        help = 'path to scales data for input sequences',
                        type = str)
    parser.add_argument('-a', '--abundance',
                        help = 'path to abundance data for input sequences',
                        type = str)
    parser.add_argument('-kv', '--knownvalid',
                        help = 'path to file listing known valid input'
                               'sequences',
                        type = str)
    parser.add_argument('-ki', '--knowninvalid',
                        help = 'path to file listing known invalid input'
                               'sequences',
                        type = str)
    parser.add_argument('-v', '--validscales',
                        help = 'path to scales data for known valid sequences',
                        type = str)
    parser.add_argument('-i', '--invalidscales',
                        help = 'path to scales data for known invalid sequences',
                        type = str)
    parser.add_argument('-z', '--trainsize',
                        help = 'maximum size of the training dataset, if the'
                               'total number of invalid and valid data points'
                               'is greater than this, it will be randomly '
                               'subset to this value, retaining equal '
                               'proportions of invalid and valid data if '
                               'possible',
                        type = int, default = None)
    parser.add_argument('-o', '--output',
                        help = 'file prefix to write outputs',
                        type = str)
    
    args = parser.parse_args(arglist) if arglist else parser.parse_args()
    
    # Checking
    #parser.error
    
    sys.stderr.flush()
    return(args)

# Main

def main():
    
    args = getcliargs()
    #args = getcliargs('-s testdata/amm/amm_scales.csv -a testdata/amm/amm_reads_asv_map_rn.tsv -kv testdata/amm/amm_match.txt -ki testdata/amm/amm_lengthvar.txt -v testdata/MIDORI/MIDORI418subset_protscale_rand500.csv -o testdata/amm/prepped.pickle'.split(' '))
    
    # Process scale data
    train, cls, new = process_scaledata(args)
    
    print(f"\nLoaded protein scale data for {len(train.index)} known valid or "
          f"invalid data points and {len(new.index)} target data points to "
           "classify.")
    
    # Add abundance data to new data
    new = process_abundancedata(args.abundance, new)
    
    # Parse known sequences
    ttrain, tcls, new = process_known(args, new)
    print(f"\nIdentified {len(tcls)} pre-classified data points from target "
           "data based on known valid/invalid lists, of which "
           f"{countcls(tcls, 1)} are valid and {countcls(tcls, 0)} are invalid.")
    
    # Merge training data
        # Generate source list
    src = ['r'] * len(cls) + ['n'] * len(tcls)
        # Merge data and class list
    train = pandas.concat([train, ttrain], axis = 0, join = 'outer')
    trainindex = train.index
    traincolumn = train.columns
    cls.extend(tcls)
        # Merge source and class list for later stratification
    strat = [s + str(c) for s, c in zip(src, cls)]
    
    print(f"\nFinal data composition: {len(train.index)} training data points "
          f"({countcls(cls, 0)} invalid, {countcls(cls, 1)} valid), "
          f"{len(new.index)} target data points to classify.")
    
    # Standardisation
        # https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
        # Fit the standardisation on the real values and variation, before
        # imputation of missing values
    slscaler = preprocessing.RobustScaler().fit(train)
    
    # Impute missing values
        # https://scikit-learn.org/stable/modules/impute.html#univariate-feature-imputation
        # Using simple univariate imputation because no a priori reason that 
        # missing values (abundance) should have any relationship with 
        # values of scale
    imp = impute.SimpleImputer(missing_values = float('NaN'), 
                               strategy = 'median')
    train = imp.fit_transform(train)
    
    # Standardise data 
    train_scaled = slscaler.transform(train)
    new_scaled = slscaler.transform(new)
    
    train_scaled = pandas.DataFrame(train_scaled, 
                                    index = trainindex, columns = traincolumn)
    train_scaled.insert(0, "stratum", strat)
    train_scaled.insert(0, "class", cls)
    new_scaled = pandas.DataFrame(new_scaled, 
                                  index = new.index, columns = new.columns)
    
    # Output
    train_scaled.to_csv(f"{args.output}_trainingdata.csv", 
                        index_label = 'seqname')
    new_scaled.to_csv(f"{args.output}_newdata.csv",
                      index_label = 'seqname')
    
    print("\nMissing data imputed, all data standardised, prepared data "
          f"written to {args.output}*")
    
    


if __name__ == "__main__":
    main()
    exit()
