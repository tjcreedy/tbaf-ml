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
The protscale.csv file supplied in this repository details 54 protein scales, each of which assigns each amino acid a value. The `applyscales.py` script supplied reads a fasta file, translates the DNA sequences to amino acids (given a reading frame and translation table, which for our example dataset is frame 2 and 5), assigns the values and calculates the mean for each of these scales. We can optionally also split each sequence into a number of equally sized chunks and compute the mean of each scale for each chunk. The result is output in csv format to the standard output. Note that here we calculate the scale means for 7 chunks as well as the overall sequence. We take advantage of the optional function to calculate the correlation coefficient between all pairwise combinations of scales and remove any that highly correlate. We also take advantage of multithreading to perform the scale assessment more quickly. 
```
applyscales.py --scales protscale.csv --readingframe 2 --maxcorrelation 0.95 --threads 10 --table 5 --chunks 7 < references_final.fasta > references_protscale.csv
```

This data is now ready to be prepared for use in training the model. The above three steps should be run again for any other reference data, if separate.

## Step 2: Preparing the focal dataset
In order to reduce the total data volume and remove easily-identifiable spurious sequences, it's strongly recommended to run denoising first, or at least remove all singleton ASVs. I also suggest removing ASVs that are  more than 3 bases longer or shorter than the expected length. In order to train the model, we must identify some sequences from the focal dataset that are valid or spurious, so I do not recommend any stricter length filtering nor any filtering based on translation. I assume the ASVs to be filtered are in a file called `ASVs.fasta`

### Identify valid sequences
If a reference set of sequences is available (here called `refdb.fasta`), BLAST the ASVs against this reference set and parse the output to retrieve a list of the ASVs that match. Generally this should be fairly strict and retain only very high likelihood matches.
```
blastn -query ASVs.fasta -db refdb.fasta -evalue 0.001 -perc_identity 100 -num_threads 10 -outfmt 6 -out ASVs_v_refdb.blast.txt
```
We filter the matches to find only those that match at sufficient length
```
awk '$4 > 380' ASVs_v_refdb.blast.txt | cut -f1 | sort | uniq > ASVs_refmatch.txt
```
The same approach could be used with a local version of GenBank if desired. It is strongly suggested that only very very high likelihood matches be retained. This step is optional
```
blastn -query ASVs.fasta -db /path/to/nt -evalue 0.001 -perc_identity 100 -num_threads 10 -outfmt 6 -out ASVs_v_nt.blast.txt
cat ASVs_v_refdb.blast.txt ASVs_v_nt.blast.txt | awk '$4 > 380' | cut -f1 | sort | uniq > ASVs_refmatch.txt
```
Clean up
```
rm ASVs_v_refdb.blast.txt ASVs_v_nt.blast.txt
```

### Identify spurious sequences
I assume that any ASVs that vary from the target length are likely errors. However, length variants due to insertions or deletions will have frame shifts in translation, so these must be aligned, the insertions removed and the deletions patched in order to ensure that these variants do not introduce unrealistic variation into the training data. Here, the target length is exactly 418bp.
First split the data into 418bp and all other, recording the names of the variants
```
vsearch --fastx_filter ASVs.fasta --fastq_minlen 418 --fastq_maxlen 418 --fastaout ASVs_418.fasta --fastaout_discarded ASVs_lengthvar.fasta
grep -oP "(?<=^>).*$" ASVs_lengthvar.fasta > ASVs_lengthexclude.txt
```
Take a subset of 2000 ASVs of the target length and align the variants to this. The alignment command includes the argument `--keeplength` which trims all insertions caused by the added sequences, so the resulting alignment length is the same as the target length. We also replace all resulting gaps in the alignment with `N`s.
```
perl -pe '$. > 1 and /^>/ ? print "\n" : chomp' ASVs_418.fasta | head -n 4000 > ASVs_418_2k.fasta # The perl oneliner unwraps the sequences first to easily subset
mafft --add ASVs_lengthvar.fasta --op 5 --ep 1 --keeplength --thread 30 ASVs_418_2k.fasta | sed -e "/^[^>]/s/-/N/g" > ASVs_lengthvar.aln.fasta
```
Concatenate the modified length variants to the target ASVs
```
perl -pe '$. > 1 and /^>/ ? print "\n" : chomp' ASVs_lengthvar.aln.fasta | tail -n +4001 > ASVs_lengthvar_modified.fasta
cat ASVs_418.fasta ASVs_lengthvar_modified.fasta > ASVs_MLinput.fasta
```
Clean up
```
rm ASVs_lengthvar* ASVs_418*
```

### Calculate protein scale data
This is performed in exactly the same way as for the reference data above. It is important that this have identical parameters for both reference and focal data.
```
applyscales.py --scales protscale.csv --readingframe 2 --maxcorrelation 0.95 --threads 10 --table 5 --chunks 7 < ASVs_MLinput.fasta > ASVs_protscale.csv
```

### Calculate read abundance data
Here we calculate the number of reads of each ASV in each of the samples within the metabarcoding dataset as a whole. The abundance profile of each ASV is used to better characterise the sequences for the classification.
This assumes that you have a single large fasta file called `reads.fasta` which contains all of the reads for your project, with the sample that each read belongs to identified by a `;sample=XXXX;` tag in the header of each read. 

Note: the inputs here are `ASVs.fasta`, not `ASVs_MLinput.fasta`. Using the latter will generate erroneous data!
```
vsearch --search_exact reads.fasta -db ASVs.fasta -otutabout reads_ASVs_map.tsv
```
If your ASVs have any annotations (e.g. `;size=YYY;`) in their headers, this data will be removed by `search_exact`, so we must return it in order for subsequent steps to accurately match up data.
```
grep -oP "(?<=^>).*$" ASVs.fasta > ASVs_names.txt
cut -f1 reads_ASVs_map.tsv | sed "s/$/;/" | xargs -Iqname grep qname ASVs_names.txt | cat <(echo "#OTU ID") /dev/stdin | paste /dev/stdin <(cut -f2- reads_ASVs_map.tsv) > reads_ASVs_map_final.tsv
```

## Step 3: Merging data in preparation for classification

All the above data is parsed, merged together, missing values imputed, and rescaled to form the datasets for classification using the `prepml.py` script. 
This command includes the optional `--usestopcount`, whereby if the translation of an amino acid contains stop codons, this sequence will be recorded as invalid. The optional argument `--addsize` includes the total read count of an ASV (from a ;size= header annotation) as an additional feature in the dataset. The argument `--dispersion 0.01` removes any features that have a dispersion (=variance/mean) of less than 0.01 in the training data.
```
prepml.py --scales ASVs_protscale.csv --abundance reads_ASVs_map_final.tsv --knownvalid ASVs_refmatch.txt --knowninvalid ASVs_lengthexclude.txt --validscales references_protscale.csv --usestopcount --addsize --dispersion 0 --output prepped
```
This command will output two `.csv` tabular files containing the standardised data in the correct format for training and classification, one each for the training data (known valid or spurious) and the new data (ASVs to be classified into spurious or valid). The training data will have two more columns than the new data, for the "class" of each training data point (0 = spurious, 1 = valid) and for the "stratum" of each training datapoint (r0 = spurious from the reference data, n1 = valid from the new data, etc). 

## Step 4: Training the classifier

Training is run using the `train.py` script, using the data in the `prepped_trainingdata.csv` file produced in the previous step. The training of the linearSVC model is performed using cross-validation to tune the `C` and `tol` hyperparameters to generate the best model under three scoring systems.


Simply, rather than just training the classifier on the entire training dataset without validation, the training data is repeatedly split into training and test data. For each combination of hyperparameters, the classifier trains on a 'training' subset of the overall training data, with each training run validated against left-out 'test' data to score the average accuracy, precision and recall of the estimator. The best performing classifier model and hyperparameters for each score is tested again against an overall train-test split, and comprehensive score data is output to a csv and a pdf for inspection. The best three classifier models are output for use in final scallification.
```
train.py -data prepped_trainingdata.pickle -threads 15 --maxiter 5000 --output results
```
The `--maxiter` parameter is passed to linearSVC - if the classifier fails to converge with this number of iterations, warnings will print. This may mean that optimal hyperparameters are not being achieved with the current maximum iterations, and you may want to consider running again with a larger value. The default is 1000.

Three scores - accuracy, precision and recall - are computed and analysed, and the performance of the classifier should be carefully inspected with respect to these scores. The scores are computed from the results of running a given model on known data, and are a functions of rate of false positives and/or false negatives. Accuracy measures the overall accuracy, viz. the proportion of correct classifications. Precision is the proportion of all positive classifications that are correct, i.e. higher values of precision denote lower rates of false positives. Recall is the proportion of positive cases that were correctly classified, i.e. higher values of recall denote lower rates of false negatives. The analysis of the best-scoring model for each score is output to `results.pdf`. The score used to select the best model should be based on the downstream research questions. For example, the recall score should be used if maximal preservation of valid ASVs is desired, while precision should be used for maximal removal of spurious ASVs. Accuracy provides an overall balance between the two. It should of course be remembered that the statistics presented in the pdf pertain only to the training data, and are only estimates of the performance of the classifier on the unknown novel data.

The script will output a \_bestestimators.pickle file which contains the best trained model for each score. This can be applied to new data in the next step.

## Step 4: Classifying new data

This step brings together data generated by previous steps to classify any unknown input ASVs and output the set of ASVs that are 'valid', either through prior determination or through novel classification using an model trained in the previous step. 
Here we will be using the best estimator according to the accuracy score.
```
classify.py --estimators results_bestestimators.pickle --score accuracy --asvs ASVs_MLinput.fasta --newdata prepped_newdata.csv --trainingdata prepped_trainingdata.csv --output classified
```
Two or three files will be output:
classified_valid.fasta will contain all sequences that were determined to be valid in Step 2, plus any sequences the selected model classified as valid according to the protein scale and abundance data
classified_invalid.fasta will contain all sequences that were determined to be spurious in Step 2, plus any sequences the selected model classified as invalid accordin to the protein scale and abundance data
classified_unknown.fasta will contain any sequences that occur in the file supplied to `--asvs` but that don't occur in `prepped_newdata.csv` or `prepped_trainingdata.csv`.
