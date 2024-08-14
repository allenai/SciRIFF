# Unzip the evidence inference dataset.
# Run this script from the `tasks` directory.

# Exit if the dataset's already there.
if [ -e data/preprocessing/processed/evidence_inference ]
then
    echo "Evidence inference dataset already found; exiting."
    exit 0
fi

unzip data/lfs_data/evidence_inference_linearized.zip -d data/preprocessing/downloads/evidence_inference
rm -r data/preprocessing/downloads/evidence_inference/__MACOSX

# The unzipped data is a .csv; format it as json.
python script/data/process_evidence_inference.py
