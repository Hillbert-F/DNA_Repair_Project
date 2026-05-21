#!/bin/bash
set -euo pipefail

export PDBid=$1
protChain="A"

find . -type f -name "proteinList.txt" -not -path "./proteinList.txt" -exec cp ./proteinList.txt {} \;
cp PDBs/${PDBid}_modified.pdb native_structures_pdbs_with_virtual_cbs/native.pdb

cp proteins_list.txt sequences/
cp native_structures_pdbs_with_virtual_cbs/native.pdb sequences/
cd sequences/
python buildseq.py native
cp input_sequences.seq dna.seq
python mapDNAseq_reverse.py dna.seq dna_modeller.seq
python combine_DNAPro.py
export cutoff=1.2
python find_cm_residues.py native.pdb $cutoff randomize_position_prot.txt randomize_position_DNA.txt

rm -rf DNA_randomization
mkdir -p DNA_randomization
cp randomize_position_DNA.txt native.seq native.decoys DNA_randomization/

rm -rf CPLEX_randomization
mkdir -p CPLEX_randomization
cat DNA_randomization/native.decoys > CPLEX_randomization/native.decoys

cd ../
grep "CA\|O5'" native_structures_pdbs_with_virtual_cbs/native.pdb > tmp.txt
tot_resnum=$(grep '^ATOM' tmp.txt | wc -l)
python create_tms.py sequences/DNA_randomization/randomize_position_DNA.txt $tot_resnum $PDBid
sed "s/CPLEX_NAME/$PDBid/g; s/PROT_CHAIN/$protChain/g" template_evaluate_phi.py > evaluate_phi.py
python evaluate_phi.py
