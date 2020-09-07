# tbaf-ml
Translation-Based Amplicon Filtering using Machine Learning

The following pipeline describes a method for identifying real or spurious Amplicon Sequence Variants (ASVs) produced through metabarcoding of a coding locus. This pipeline treats the identification of ASVs as real or spurious as a classification problem and applies a Machine Learning approach, using a Linear Support Vector Classifier (linearSVC). This pipeline is intended as a complement to other available methods (e.g. denoising with UNOISE or DADA2, or abundance filtering with metaMATE), however is not as developed as these one-stop tools. This is partly due to lack of time availability to write this up into a neat tool, and partly due to a desire to ensure that the Machine Learning approach is not applied without sufficient consideration of all necessary training and testing of the linearSVC estimator. 

Machine Learning is implemented through the python scikit-learn library, and it is strongly suggested that careful study is made of the documentation and references there cited (https://scikit-learn.org/stable/modules/svm.html#svm-classification).

The classification task is based upon two main sets of data about the ASVs, both widely underused for filtering spurious sequences. The first is a set of summary statistics based on protein scale values calculated from the amino acid translation of the ASVs. The second is ASV read abundances across subsets of the dataset, generally samples. While total ASV abundance in the dataset is widely used in filtering and denoising, the pattern of ASV abundance at the sample level may allow for more sensitive filtering approaches, as implemented in metaMATE.

Training of the linearSVC model is performed using a training set comprised of:
  1. External sequences, such as those from global databases, known to be valid
  2. A subset of the dataset in question identified as valid or invalid due to a) matches against a reference set or b) length variation or the presence of stop codons, respectively.
  
# The pipeline

The pipeline requires python >=3.6, the python scripts in this repository, and that the `sklearn`, `numpy`, `scipy` and `pandas` libraries be installed.

Also required are the `blast`, `vsearch` and `cutadapt` for other parts of the pipeline.

This pipeline is based on a CO1 amplicon, specifically the 418bp region between Ill_B_F and Fol_degen_rev. It will likely work fine on other coding loci, particularly mitochondrial loci, but hasn't explicitely been tested elsewhere. It is unlikely to be as effective for non-coding loci.

## Step 1: External data

It is strongly suggested for maximum precision and accuracy that outside reference data is used in training the model. The MIDORI UNIQUE dataset was used for developing this pipeline, selecting only those sequences belonging an appropriate taxonomic group. 

Here we use `references.fasta` to refer to this data, note that if you have both valid and invalid reference data, some steps should be run twice, once for each.

### Extract the region of interest from the reference dataset
We use cutadapt to perform in-silico PCR using the primers of our target region on all sequences in `references.fasta`. Given that many standard CO1 barcodes will not include both primer regions, only one primer must be present for a sequence to be retained. Here cutadapt uses the default mismatch rate of 10%.
```
cutadapt -a "CCNGAYATRGCNTTYCCNCG;optional...TGRTTYTTYGGNCAYCCNGARGTNTA;optional" --discard-untrimmed -o references_amplicons.fasta references.fasta
```
### Retain only reference amplicons of the precise target length
Because in silico PCR was relatively lax, many sequences will be retained that are shorter or longer than the target region. For fair comparison to the dataset to filter, we retain only that are exactly the right length, in this case 418bp
```
vsearch --fastx_filter references_amplicons.fasta --fastq_minlen 418 --fastq_maxlen 418 --fastaout references_final.fasta
```

### Calculate protein scale values
The protscale.csv file supplied in this repository details 54 protein scales, each of which assigns each amino acid a value. The `applyscales.py` script supplied reads a fasta file, translates the DNA sequences to amino acids (given a reading frame and translation table, which for our example dataset is frame 2 and 5), assigns the values and calculates the mean for each of these scales. The result is output in csv format to the standard output. Note that here we take advantage of the optional function to calculate the correlation coefficient between all pairwise combinations of scales and remove any that highly correlate. We also take advantage of multithreading to perform the scale assessment more quickly. 
```
python3 applyscales.py --scales protscale.csv --readingframe 2 --maxcorrelation 0.95 --threads 10 --table 5 < references_final.fasta > references_protscale.csv
```

This data is now ready to be prepared for use in training the model. The above three steps should be run again for any other reference data, if separate.

## Step 2: Preparing the focal dataset

In order to train the model, we must identify some sequences from the focal dataset that are valid or spurious. For this reason, I suggest that this pipeline be run after any denoising performed on the dataset (to reduce data volume), but before any length or translation filtering (so that identifiable spurious sequences exist). I assume the ASVs to be filtered are in a file called `ASVs.fasta`

### Identify valid sequences
If a reference set of sequences is available (here called `refdb.fasta`), BLAST the ASVs against this reference set and parse the output to retrieve a list of the ASVs that match. Generally this should be fairly strict and retain only very high likelihood matches.
```
blastn -query ASVs.fasta -db refdb.fasta -evalue 0.001 -perc_identity 100 -num_threads 10 -outfmt 6 -out ASVs_v_refdb.blast.txt
```
We filter the matches to find only those that match at sufficient length
```
awk '$4 > 380' ASVs_v_refdb.blast.txt | cut -f1 | sort | uniq > ASVs_refmatch.txt
rm ASVs_v_refdb.blast.txt
```

### Identify spurious sequences
I assume that any ASVs that vary from the target length by anything other than one complete codon are likely errors, i.e. the permitted sequence lengths are 415, 418 or 421 base pairs. 
```
for l in 415 418 421; do vsearch --fastx_filter ASVs.fasta --fastq_minlen $l --fastq_maxlen $l --fastaout lengthcheck_$l.fasta; done
cat lengthcheck_*.fasta | grep "^>" > lengthcheck_potentials.txt
diff --new-line-format="" --unchanged-line-format="" <(grep "^>" ASVs.fasta | sort) lengthcheck_potentials.txt | sed -e "s/^>//" > ASVs_lengthexclude.txt
rm lengthcheck_*
```

### Calculate protein scale data
This is performed in exactly the same way as for the reference data above
```
python3 applyscales.py --scales protscale.csv --readingframe 2 --maxcorrelation 0.95 --threads 10 --table 5 < ASVs.fasta > ASVs_protscale.csv
```

### Calculate read abundance data
Here we calculate the number of reads of each ASV in each of the samples within the metabarcoding dataset as a whole. The abundance profile of each ASV is used to better characterise the sequences for the classification.
This assumes that you have a single large fasta file called `reads.fasta` which contains all of the reads for your project, with the sample that each read belongs to identified by a `;sample=XXXX;` tag in the header of each read. 
```
vsearch --search_exact reads.fasta -db ASVs.fasta -otutabout reads_ASVs_map.tsv
```
If your ASVs have any annotations (e.g. `;size=YYY;`) in their headers, this data will be removed by `search_exact`, so we must return it in order for subsequent steps to accurately match up data.
```
grep -oP "(?<=^>).*$" ASVs.fasta > ASVs_names.txt
cut -f1 reads_ASVs_map.tsv | sed "s/$/;/" | xargs -Iqname grep qname ASVs_names.txt | cat <(echo "#OTU ID") /dev/stdin) | paste /dev/stdin <(cut -f2- reads_ASVs_map.tsv) > reads_ASVs_map_final.tsv
```

## Step 3: Merging data in preparation for classification

All the above data is parsed, merged together, missing values imputed, and rescaled to form the datasets for classification using the `prepml.py` script. 
```
python3 prepml.py --scales ASVs_protscale.csv --abundance reads_ASVs_map_final.tsv --knownvalid ASVs_refmatch.txt --knowninvalid ASVs_lengthexclude.txt --validscales references_protscale.csv --output prepped
```
This command will output two `.csv` tabular files containing the standardised data in the correct format for training and classification, one each for the training data (known valid or spurious) and the new data (ASVs to be classified into spurious or valid). The training data will have two more columns than the new data, for the "class" of each training data point (0 = spurious, 1 = valid) and for the "stratum" of each training datapoint (r0 = spurious from the reference data, n1 = valid from the new data, etc). 

## Step 4: Training the classifier

Training is run using the `exploreLinearSVC.py` script, using the data in the `_trainingdata.csv` file produced in the previous step. The training of the linearSVC model is performed using cross-validation to tune the `C` and `tol` hyperparameters to generate the best model under three scoring systems. An initial range of hyperparameter values is 


Simply, rather than just training the classifier on the entire training dataset without validation, the training data is repeatedly split into training and test data. For each combination of hyperparameters, the classifier trains on a 'training' subset of the overall training data, with each training run validated against left-out 'test' data to score the average accuracy, precision and recall of the estimator. The best performing classifier model and hyperparameters for each score is tested again against an overall train-test split, and comprehensive score data is output to a csv and a pdf for inspection. The best three classifier models are output for use in final scallification.
```
python3 exploreLinearSVC.py -data prepped.pickle -threads 10 --maxiter 5000 --output results
```
The `--maxiter` parameter is passed to linearSVC - if the classifier fails to converge with this number of iterations, warnings will print. This may mean that optimal hyperparameters are not being achieved with the current maximum iterations, and you may want to consider running again with a larger value. The default is 1000.

Three scores - accuracy, precision and recall - are computed and analysed, and the performance of the classifier should be carefully inspected with respect to these scores. The scores are computed from the results of running a given model on known data, and are a functions of rate of false positives and/or false negatives. Accuracy measures the overall accuracy, viz. the proportion of correct classifications. Precision is the proportion of all positive classifications that are correct, i.e. higher values of precision denote lower rates of false positives. Recall is the proportion of positive cases that were correctly classified, i.e. higher values of recall denote lower rates of false negatives. The analysis of the best-scoring model for each score is output to `results.pdf`. The score used to select the best model should be based on the downstream research questions. For example, the recall score should be used if maximal preservation of valid ASVs is desired, while precision should be used for maximal removal of spurious ASVs. Accuracy provides an overall balance between the two. It should of course be remembered that the statistics presented in the pdf pertain only to the training data, and are only estimates of the performance of the classifier on the unknown novel data.





