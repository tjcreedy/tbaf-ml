#!/usr/bin/env bash

asvs=$1
len=$2

name=${asvs%.fa*}

echo -e  "\nSeparating sequences of the target length and others\n"
vsearch --fastx_filter $asvs --fastq_minlen $len --fastq_maxlen $len --fastaout ${name}_temp_t --fastaout_discarded ${name}_temp_lv
echo -e  "\nGetting the names of the sequences not of the target length\n"
grep -oP "(?<=^>).*$" ${name}_temp_lv > ${name}_lengthexclude.txt
nl=$(grep -c "^>" ${name}_temp_t)
if [[ "$nl" -gt 2000 ]];
then
	echo -e  "\nLinearising the sequences of the target length, then taking the first 2000 sequences\n"
	perl -pe '$. > 1 and /^>/ ? print "\n" : chomp' ${name}_temp_t | head -n 4000 > ${name}_temp_ts
else
	cp ${name}_temp_t ${name}_temp_ts
fi
echo -e  "\nDoing alignment\n"
mafft --add ${name}_temp_lv --op 5 --ep 1 --keeplength --thread 30 ${name}_temp_ts | sed -e "/^[^>]/s/-/N/g" > ${name}_temp_aln
nN=$(grep "^[^>]" ${name}_temp_aln | grep -c "N")
echo -e "\nInserted $nN Ns into alignment gaps\n"
echo -e  "\nConcatenating all sequences together\n"
perl -pe '$. > 1 and /^>/ ? print "\n" : chomp' ${name}_temp_aln | tail -n +4001 > ${name}_temp_lvm
cat ${name}_temp_t ${name}_temp_lvm > ${name}_MLinput.fasta
echo -e  "\nCleaning up\n"
rm ${name}_temp*
