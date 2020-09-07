#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Thomas J. Creedy
"""

# Imports

import sys
import argparse
import csv
import multiprocessing
import functools
import pandas as pd
import numpy as np 

import textwrap as _textwrap
from Bio import SeqIO
from Bio import SeqUtils
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# Global variables

AA = 'ARNDCQEGHILKMFPSTWYV'

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

# Function definitions

def to_scale_dict(line, head):
    scale = dict()
    scale['scale'] = {aa: float(n) for aa, n in zip(head, line) if aa in AA}
    if len(AA) != len(scale['scale'].keys()):
        raise KeyError(f"header does not contain all of {AA}")
    if 'name' not in head:
        raise ValueError("header does not contain a \'name\' column")
    for n in ['name', 'type', 'description', 'reference']:
        scale[n] = line[head.index(n)] if n in head else None
    return(scale)

def correlating_scales(scales, threshold):
    # Convert to pandas df
    scaledf = pd.DataFrame({s['name']: [s['scale'][aa] for aa in AA] 
                                                              for s in scales})
    # Generate correlation matrix
    scalecm = scaledf.corr().abs()
    # Drop diagonal
    mask = np.zeros(scalecm.shape, dtype = bool)
    np.fill_diagonal(mask, 1)
    scalecm = scalecm.mask(mask)
    # Sort by max value
    cols = scalecm.max().sort_values(ascending = False).index
    scalecm = scalecm.reindex(cols, axis = 1)
    scalecm = scalecm.reindex(cols, axis = 0)
    # Drop scales
    drop = []
    for i in scalecm.columns:
        if i in drop:
            continue
        rcols = [c for c in scalecm.columns if c not in drop]
        if scalecm[rcols].loc[i].max() < threshold:
            continue
        for j in rcols:
            if j == i:
                continue
            #i, j = scalecm.columns[0:2]
            if scalecm[i].loc[j] > threshold:
                rcols = [c for c in scalecm.columns if c not in drop]
                means = [scalecm[rcols].loc[c].mean() for c in [i, j]]
                todrop = [i, j][means.index(max(means))]
                drop.append(todrop)
                if todrop == i:
                    break
    return(drop)

def parse_scales(path, retainthreshold):
    #path, retainthreshold = args.scales, 0.95
    # Parse the file
    with open(path, 'r') as fh:
        reader = csv.reader(fh)
        head = next(reader)
        scales = [to_scale_dict(line, head) for line in reader]
    
    if retainthreshold < 1:
        acscales = correlating_scales(scales, retainthreshold)
        scales = [s for s in scales if s['name'] not in acscales]
    
    return(scales)

#def chunker(seq, n):
#    for i in range(0, len(seq), int(n)):
#        yield(seq[i:i+n])

def scale_evaluation(seqr, aastr, scales, chunks = 1):
    #seqr, aastr = seqr.seq, seqstring
    # Summary information
    out = {'name': seqr.id,
           'n_stops': aastr.count('*'),
           'n_nucs': len(seqr.seq)}
    # GC percentages
    for k, v in zip(['total', 'pos1', 'pos2', 'pos3'], 
                    SeqUtils.GC123(seqr.seq)):
        out[f"GC_pc_{k}"] = v
    
    def _part_eval(aapart, i = None):
        sfx = '' if i == None else f"_chunk{i}"
        px = ProteinAnalysis(aapart)
        aalen = len(aapart)
        for k, v in px.get_amino_acids_percent().items():
            out[f"prop_{k}_AA{sfx}"] = v
        for scale in scales:
            svs = sum([scale['scale'][aa] for aa in aapart if aa in AA])
            out[scale['name'] + sfx] = svs/aalen
    
    # Protein data 
    if chunks == 1:
        _part_eval(aastr)
    else:
        for i, aapart in enumerate(np.array_split(list(aastr), chunks)):
            _part_eval(''.join(aapart), i)
    return(out)

def write_out(scales, prinq):
    outlist = []
    while 1:
        queueitem = prinq.get()
        if queueitem is None: break
        outlist.append(queueitem)
    outdf = pd.DataFrame(outlist)
    outdf.set_index('name', inplace = True)
    outdf.sort_index(axis = 1, inplace = True)
    outdf.to_csv(sys.stdout, index = True)

def process_seqrecord(scales, prinq, args, rf0, seqr):
    #seqr = next(SeqIO.parse('testdata/amm/amm_asvs.fasta', 'fasta'))
    aastr = str(seqr[rf0:].translate(table = args.table).seq)
    prinq.put(scale_evaluation(seqr, aastr, scales, args.chunks))

def getcliargs(arglist = None):
    
    parser = argparse.ArgumentParser(description="""
        description:
        This script calculates amino acid scale values for supplied sequences.
        Supply one or more nucleotide sequences in fasta format on STDIN, and
        the path to a csv of protein scales to -s/--scales. All nucleotide
        sequences must share the same NCBI translation table and the same
        reading frame.
        |n
        The protein scale csv must be formatted as follows. The first line 
        must be column headings that include 'name' and every amino acid in 
        capitalised single-letter format, and may also include 'type', 
        'description' and 'reference', which are parsed but not currently used.
        Columns may be in any order, and any other columns are permitted but
        will be ignored. Each row should be a different scale with a unique 
        name and amino acid values in positions corresponding to the header 
        row.
        |n
        Optionally, protein scales can be filtered for high pairwise 
        correlations to remove redundancy in the dataset. For any pairwise
        correlations greater than the value given to -m/--maxcorrelation, 
        the scale with the greater mean correlation is dropped.
        """,formatter_class=MultilineFormatter)
    
    parser._optionals.title = "arguments"
    
    parser.add_argument('-s', '--scales', metavar = 'path',
                        help = 'path to a tabular csv file containing protein '
                               'scale values',
                        type = str, required = True)
    parser.add_argument('-m', '--maxcorrelation',
                        help = 'value of pairwise correlation above which '
                               'highly correlating scales will be removed',
                        type = float, default = 1.0,
                        action = Range, minimum = 0, maximum = 1)
    parser.add_argument('-rf', '--readingframe',
                        help = 'an integer denoting the position of the '
                               'input sequences on which to begin '
                               'translation',
                        type = int, choices = [1, 2, 3], required = True)
    parser.add_argument('-c', '--chunks',
                        help = 'if supplied, amino acid sequences will be '
                               'split into this number of equally-sized '
                               'segments of the given length and scale '
                               'values computed separately and averaged '
                               'on each chunk',
                        type = int, default = 1)
    parser.add_argument('-t', '--threads',
                        help = 'number of parallel threads',
                        type = int, default = 1)
    parser.add_argument('-b', '--table',
                        help = 'an integer denoting the NCBI translation '
                               'table to use for amino acid translation',
                        type = int, required = True,
                        action = Range, minimum = 1, maximum = 33)
    
    args = parser.parse_args(arglist) if arglist else parser.parse_args()
    
    # Checking
    #parser.error
    
    sys.stderr.flush()
    return(args)

# Main

def main():
    args = getcliargs()
    #args = getcliargs(['-s', 'protscale.csv', '-rf', '2', '-b', '5'])
    
    scales = parse_scales(args.scales, args.maxcorrelation)
    rf0 = args.readingframe-1
    
    # Initialise the queue manager and pool
    manager = multiprocessing.Manager()
    pool =  multiprocessing.Pool(args.threads + 1)
    prinq = manager.Queue()
    
    printwatch = pool.apply_async(functools.partial(write_out, scales),
                                  (prinq,))
    
    pool.map(functools.partial(process_seqrecord, scales, prinq, args, rf0),
                               SeqIO.parse(sys.stdin, 'fasta'))
    
    prinq.put(None)
    pool.close()
    pool.join()
    printwatch.get()

if __name__ == "__main__":
    main()
    exit()

