#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Thomas J. Creedy
"""

# Imports

import sys
import argparse
import pickle
import re
import warnings

import pandas as pd
import numpy as np

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

def process_scaledata(args):
    # Load new data
    new = pd.read_csv(args.scales, index_col='name')
    # Load training data if present
    train = list()
    for p in [args.validscales, args.invalidscales]:
        if p is not None:
            train.append(pd.read_csv(p, index_col='name'))
        else:
            train.append(pd.DataFrame(columns=new.columns))
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
    train = pd.concat(train, axis=0, join='inner')
    cls = [1] * lens[0] + [0] * lens[1]
    if args.usestopcount:
        cls = [0 if s > 0 else c for c, s in zip(cls, train['n_stops'])]
    return(train, cls, new)
    
def process_abundancedata(path, addsize):
    #path = args.abundance
    indf = pd.read_csv(path, sep = '\t', index_col = 0)
    # Standardise by sample totals
    indf = indf.div(indf.sum(axis = 0), axis = 1)
    # Discard sample-wise values and instead sort by magnitude
    a = indf.values
    a.sort(axis = 1)
    indf = pd.DataFrame(a[:, ::-1], indf.index)
    # Drop any columns with zero totals
    indf = indf.loc[:,indf.sum(axis = 0) > 0]
    # Add size if size labels
    if addsize:
        if all([';size=' in i for i in indf.index]):
            indf['size'] = [int(re.search("(?<=;size=)\d+", i).group()) 
                                                        for i in indf.index]
        else:
            warnings.warn("Warning: size annotations not found")
    # Merge with scales data and return
    return(indf)

def process_known(args, newscale):
    # Set up known dict
    known = {'v': [], 'i': []}
    # Read in known lists if available
    for t, p in zip('vi', [args.knownvalid, args.knowninvalid]):
        if p:
            known[t] = [x.strip() for x in open(p).readlines()]
    # Check for stops
    if args.usestopcount:
        known['i'].extend(newscale.index[newscale['n_stops'] > 0].tolist())
    # Find unique
    known['i'] = set(known['i'])
    # Remove any invalid from valid
    known['v'] = set(known['v']) - known['i']
    return(known['v'], known['i'])

def split_by_known(indf, valid, invalid):
    #indf, drop0 = newabun, True
    # Generate class list
    cls = [1] * len(valid) + [0] * len(invalid)
    # Split data frame
    trainidx = list(valid) + list(invalid)
    train = indf.loc[trainidx]
    new = indf.loc[[i for i in indf.index if i not in trainidx]]
    return(train, cls, new)

def drop_dispersion(train, new, disthresh):
    #train, new, disthresh = trainscale, newscale, args.dispersion
    dispersion = train.var()/abs(train.mean(axis = 0))
    retain = [c for c, d in zip(train, dispersion) if d > disthresh]
    return(train[retain], new[retain], len(train.columns) - len(retain))

def countcls(cls, v):
    if type(v) is int:
        return(len([c for c in cls if c == v]))
    else:
        return([len([c for c in cls if c == vi]) for vi in v])

def scale_pddf(scaler, df):
    return(pd.DataFrame(scaler.transform(df), 
                        index = df.index, 
                        columns = df.columns))

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
                        help = 'path to scales data for known invalid '
                               'sequences',
                        type = str)
    parser.add_argument('-u', '--usestopcount',
                        help = 'if a sequence reported > 0 stop codons, mark'
                               'as invalid',
                        action = 'store_true')
    parser.add_argument('-d', '--addsize',
                        help = 'if the sequence names in --abundance have '
                               ';size=YYYY annotations, add this data',
                        action = 'store_true'),
    parser.add_argument('-p', '--dispersion',
                        help = 'drop all features (columns) that do not exceed'
                               'the given dispersion (=var/mean)',
                        default = 0, type = float)
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
    #args = getcliargs('-s testdata/amm/amm_protscale_7chunks.csv -a testdata/amm/amm_reads_asv_map_rn.tsv -kv testdata/amm/amm_match.txt -ki testdata/amm/amm_lengthvar.txt -v testdata/MIDORI/MIDORI418_Insecta_rand500_protscale_7chunks.csv -o testdata/amm/prepped -u -d -p 0.001'.split(' '))
    
    print("\nLoading scale data...",)
    
    # Process scale data
    trainscale, cls, newscale = process_scaledata(args)
    
    print(f"\rLoaded protein scale data for {len(trainscale.index)} known "
          f"valid or invalid data points and {len(newscale.index)} target "
          f"data points")
    
    # Add abundance data to new data
    newabun = process_abundancedata(args.abundance, args.addsize)
    nabun = len(newabun.columns) - (1 if args.addsize else 0)
    print(f"\nLoaded {nabun} abundance columns"
          f"{' plus ASV sizes parsed from ASV names' if args.addsize else ''}")
    
    # Parse known sequences and split up new data
    valid, invalid =  process_known(args, newscale)
    ntrainscale, ncls, newscale = split_by_known(newscale, valid, invalid)
    ntrainabun, ncls, newabun = split_by_known(newabun, valid, invalid)
    print(f"\nIdentified {len(ncls)} pre-classified data points from target "
           "data based on known valid/invalid lists, of which "
          f"{countcls(ncls, 1)} are valid and {countcls(ncls, 0)} are invalid.")
    
    # Merge together protein scale training data
    trainscale = pd.concat([trainscale, ntrainscale], 
                               axis = 0, join = 'outer')
    src = src = ['r'] * len(cls) + ['n'] * len(ncls)
    cls = cls + ncls
    strat = [s + str(c) for s, c in zip(src, cls)]
    
    print(f"\nCompiled scale data has {len(trainscale.columns)} columns")
    
    # Drop columns that do not vary in training data
    trainscale, newscale, nsd = drop_dispersion(trainscale, newscale, 
                                                args.dispersion)
    trainabun, newabun, nad = drop_dispersion(ntrainabun, newabun,
                                              args.dispersion)
    
    if nsd > 0 or nad > 0:
        print(f"\nDropped {nsd} columns from scale data and {nad} columns "
              f"from abundance data for not exceeding {args.dispersion} "
               "dispersion in the training data")
    
    # Fit the scalers pre-imputation of missing data
    scalesscaler = preprocessing.StandardScaler()
    scalesscaler.fit(trainscale)
    
    abunscaler = preprocessing.PowerTransformer()
    abunscaler.fit(trainabun)
    
    # Fill missing data in trainabun
    trainabun = pd.concat([pd.DataFrame(index = trainscale.index), trainabun], 
                          axis = 1, join = 'outer')
    
        # Impute missing values
        # https://scikit-learn.org/stable/modules/impute.html#univariate-feature-imputation
        # Using simple univariate imputation because no a priori reason that 
        # missing values (abundance) should have any relationship with 
        # values of scale
    imp = impute.SimpleImputer(missing_values = np.nan, strategy = 'median')
    imp.fit(trainabun)
    trainabun = pd.DataFrame(imp.transform(trainabun), 
                             index = trainabun.index, 
                             columns = trainabun.columns)
    
    # Scale
        # Training data
    scaledtrainabun = scale_pddf(abunscaler, trainabun)
    scaledtrainscale = scale_pddf(scalesscaler, trainscale)
        # New data
    scalednewabun = scale_pddf(abunscaler, newabun)
    scalednewscale = scale_pddf(scalesscaler, newscale)
    
    # Merge the data
    train = pd.concat([scaledtrainscale, scaledtrainabun], 
                      axis = 1, join = 'outer')
        # Check all([ts == t for ts, t in zip(trainscale.index, train.index)])
    
    new = pd.concat([scalednewscale, scalednewabun],
                    axis = 1, join = 'outer')
    
    print( "\nMissing data impututed, all data standardised\n\n"
          f"Final data composition: {len(train.columns)} scale and "
           "abundance features across all data. Training data has "
          f"{len(train.index)} data points, of which {countcls(cls, 0)} are "
          f"'invalid' and {countcls(cls, 1)} are 'valid'; there are "
          f"{len(new.index)} target data points to classify.\n\n"
           "Note: columns 'n_stops', 'n_nt_ambig' and 'n_aa_ambig' should not "
           "be used for training or classification as they are not independent"
           " from the methodology used to designate 'invalid' data points\n\n"
           "Writing completed tables...",)
    
    train.insert(0, "stratum", strat)
    train.insert(0, "class", cls)
    
    # Output
    train.to_csv(f"{args.output}_trainingdata.csv", index_label = 'seqname')
    new.to_csv(f"{args.output}_newdata.csv", index_label = 'seqname')
    
    print(f"\rPrepared training and novel data written to {args.output}*\n")


if __name__ == "__main__":
    main()
    exit()
