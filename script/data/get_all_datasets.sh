# Download all freely-available datasets. Run from project root like:
# bash script/data/get_all_datasets.sh.

mkdir -p data/preprocessing/downloads
mkdir -p data/preprocessing/processed

bash script/data/get_scierc.sh
bash script/data/get_chemprot.sh
bash script/data/get_genia.sh
bash script/data/get_evidence_inference.sh
bash script/data/get_annotated_materials_syntheses.sh
bash script/data/get_qasa.sh
bash script/data/get_bioasq_task11b.sh
