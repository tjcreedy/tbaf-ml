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

from collections import defaultdict

from Bio import SeqIO
from Bio import SeqUtils
from Bio.SeqUtils.ProtParam import ProteinAnalysis


# Global variables
NT = 'ATCG'
NTambigs = {'Y': 'CT', 'R': 'AG', 'W': 'AT', 'S': 'GC', 'K': 'TG','M': 'CA',
            'D': 'AGT', 'V': 'ACG', 'H': 'ACT', 'B': 'CGT', 'N': NT}
AA = 'ACDEFGHIKLMNPQRSTVWY'
AAambigs = {'B': 'ND', 'J': 'IL', 'Z': 'EQ', 'X': AA}

def makecounter(unique, ambigs):
    alluniq = list(unique) + list(ambigs.keys())
    counter = dict()
    for r in alluniq:
        if r in unique:
            counter[r] = {r: 1}
        else:
            counter[r] = {ra: 1/len(ambigs[r]) for ra in ambigs[r]}
    return(counter)

NTcounter, AAcounter = makecounter(NT, NTambigs), makecounter(AA, AAambigs)

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
    
#    for aa, vals in AAambigs.items():
#        scale['scale'][aa] = sum([scale['scale'][v] for v in vals])/2
    
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




def scale_evaluation(seqr, aastr, scales, chunks):
    # chunks = 7
    seqr.seq = seqr.seq.upper()
    
    out = {'name': seqr.id,
           'n_stops': aastr.count('*'),
           'n_nt_ambig': seqr.seq.count('N'),
           'n_aa_ambig': aastr.count('X')}
    
    def _count_resid(restr, restype, prop = True):
        #restr, counter, stdres, prop = ntstr, NTcounter, NT, True
        if restype == 'NT':
            counter, stdres = NTcounter, NT
        elif restype == 'AA':
            counter, stdres = AAcounter, AA
        counts = {r: 0 for r in stdres}
        restr = str(restr).replace('*', '')
        for inres in restr:
            for outres, value in counter[inres].items():
                counts[outres] += value
        if prop:
            counts = {r: v/len(restr) for r, v in counts.items()}
        return(counts)
    
    def _part_eval(ntstr, aastr, chunk = None):
        # ntstr = str(seqr.seq)
        # NT counts
        rtn = _count_resid(ntstr, 'NT')
        # NT counts by codon position
        for i in range(3):
            residcount = _count_resid(seqr.seq[i::3], 'NT')
            rtn.update({f"{k}_pos{i+1}": v for k, v in residcount.items()})
        rtn = {f"prop_nt_{k}": v for k, v in rtn.items()}
        
        # AA counts
        aacount = _count_resid(aastr, 'AA')
        # AA scale values
        scalevals = {}
        for scale in scales:
            for aa, p in aacount.items():
                nkey = f"mean_{scale['name']}"
                vals = [scale['scale'][k] * v for k, v in aacount.items()]
                scalevals[nkey] = sum(vals)
        scalevals.update({f"prop_aa_{k}": v for k, v in aacount.items()})
        rtn.update(scalevals)
        
        if chunk is not None:
            rtn = {f"{k}_chunk{chunk}": v for k, v in rtn.items()}
        return(rtn)
    
    out.update(_part_eval(seqr.seq, aastr))
    
    if chunks > 1:
        chunked = [np.array_split(list(restr), chunks) 
                      for restr in [seqr.seq, aastr]]
        for chnk, (ntpart, aapart) in enumerate(zip(*chunked)):
            out.update(_part_eval(''.join(ntpart), ''.join(aapart), chnk))
    return(out)


def write_out(scales, prinq):
    csvwrite = csv.writer(sys.stdout, delimiter=',', quotechar='"', 
                         quoting=csv.QUOTE_MINIMAL)
    head = None
    n = 0
    while 1:
        queueitem = prinq.get()
        if queueitem is None: break
        n += 1
        if n == 1:
            head = list(queueitem.keys())
            csvwrite.writerow(head)
        csvwrite.writerow([queueitem[h] for h in head])
        sys.stderr.write(f"\rDone {n} sequences")
        sys.stderr.flush()
    sys.stderr.write("\n")
    sys.stderr.flush()

def process_seqrecord(scales, prinq, args, rf0, seqr):
    #seqr = next(SeqIO.parse('testdata/cccpmeta/4_MLinput.fasta', 'fasta'))
    end = len(seqr) - ((len(seqr) - rf0) % 3)
    aastr = str(seqr[rf0:end].translate(table = args.table).seq)
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

