#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Thomas J. Creedy
"""

# Imports

import sys
import pickle
import argparse

import pandas as pd
from Bio import SeqIO

import textwrap as _textwrap

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

def required_multiple(multiple):
    class RequiredMultiple(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            if not len(values) % multiple == 0:
                msg = 'argument "{f}" requires a multiple of {multiple} values'
                msg = msg.format(f=self.dest, multiple=multiple)
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredMultiple


def getcliargs(arglist = None):
    
    parser = argparse.ArgumentParser(description="""
        description:
        |n
        Text
        |n
        Text
        """,formatter_class=MultilineFormatter)
    
    parser._optionals.title = "arguments"
    
    parser.add_argument('-e', '--estimators',
                        help = 'path to a *_bestestimators.pickle file output'
                               ' by training',
                        type = str, required = True)
    parser.add_argument('-s', '--score',
                        help = 'name of a score for which the best-optimised '
                               'estimator will be used for classification',
                        type = str, required = True)
    parser.add_argument('-a', '--asvs',
                        help = 'path to asvs to classify (may include known '
                               'valid and invalid sequences)',
                        type = str, required = True)
    parser.add_argument('-nd', '--newdata',
                        help = 'path to prepared new data',
                        type = str, required = True)
    parser.add_argument('-td', '--trainingdata',
                        help = 'path to prepared training data (used to '
                               'filter already-classified asvs)',
                        type = str, required = True)
    parser.add_argument('-o', '--output',
                        help = 'path prefix to output fastas',
                        type = str, required = True)
    
    args = parser.parse_args(arglist) if arglist else parser.parse_args()
    
    # Checking
    #parser.error
    
    sys.stderr.flush()
    return(args)


# Main

def main():
    
    args = getcliargs()
    #args = getcliargs('-e testruns/cccpmeta_rand1000_pipetest_bestestimators.pickle -s precision -a testdata/cccpmeta_rand1000/4_MLinput.fasta -nd testdata/cccpmeta_rand1000/prepped_newdata.csv -td testdata/cccpmeta_rand1000/prepped_trainingdata.csv -o testruns/cccpmeta_rand1000_pipetest_classify'.split(' '))
    
    new = pd.read_csv(args.newdata, index_col = 0)
    
    # Drop this data because this is linked identifying the known invalid data 
    dropcols = ['n_stops', 'n_nt_ambig', 'n_aa_ambig']
    new = new.drop([d for d in dropcols if d in new], axis = 1)
    
    print(f"\nLoaded {len(new.index)} total data points to classify with "
          f"{len(new.columns)} features after removal of nonindependent "
          "features")
    
    models = pickle.load(open(args.estimators, 'rb'))
    
    selectedscore = [s for s in models.keys() if args.score in s]
    
    if len(selectedscore) != 1:
        avail = ', '.join([k for k in models.keys() if k != 'fullsearch'])
        if len(selectedscore) == 0:
            err = "does not match any "
        else:
            err = "matches more than one "
        exit(f"Error supplied score to -s/--score \"{args.score}\" {err} "
             f"available scorers: {avail}")
    
    estimator = models[selectedscore[0]]
    
    print(f"\nClassifying new data using estimator with best {args.score} "
          "score")
    
    cls = estimator.predict(new)
    
    retain, reject = [], []
    for i, c in zip(new.index, cls):
        retain.append(i) if c == 1 else reject.append(i)
    
    print(f"\nClassifier identified {len(retain)} valid data points and "
          f"{len(reject)} invalid data points")
    
    train = pd.read_csv(args.trainingdata, index_col = 0)
    retain.extend(train.index[train['stratum'] == 'n1'].values)
    reject.extend(train.index[train['stratum'] == 'n0'].values)
    
    print(f"\nAfter parsing training data, total of {len(retain)} "
          f"valid sequences and {len(reject)} invalid sequences")
    
    
    print(f"\nFiltering {args.asvs}...")
    
    nretain = nreject = nunknown = 0
    
    with(open(f"{args.output}_valid.fasta", 'w'))as vfa, (
         open(f"{args.output}_invalid.fasta", 'w')) as ifa, (
         open(f"{args.output}_unknown.fasta", 'w')) as ufa:
        for seqr in SeqIO.parse(args.asvs, 'fasta'):
            if seqr.name in retain:
                nretain += 1
                vfa.write(seqr.format('fasta'))
            elif seqr.name in reject:
                nreject += 1
                ifa.write(seqr.format('fasta'))
            else:
                nunknown +=1
                ufa.write(seqr.format('fasta'))
    
    txt = ''
    if nunknown > 0:
        txt = (f". {nunknown} sequence names from {args.asvs} were not present"
               f" in {args.newdata} or {args.trainingdata}. These sequences "
               f"were written to {args.output}_unknown.fasta")
    
    print(f"\nFiltering complete. Output {nretain} valid sequences to "
          f"{args.output}_valid.fasta and {nreject} invalid sequences to "
          f"{args.output}_invalid.fasta{txt}")
    

if __name__ == "__main__":
    main()
    exit()

