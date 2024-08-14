# Process the chemprot ataset.
# Run this script from the `tasks` directory by invoking `bash script/get_chemprot.sh`

# Exit if the dataset's already there.
if [ -e data/preprocessing/processed/chemprot ]
then
    echo "Processed chemprot dataset already found; exiting."
    exit 0
fi

tar -xvf data/lfs_data/chemprot.tar.gz -C data/preprocessing/downloads

mkdir -p data/preprocessing/processed/chemprot
python script/data/process_chemprot.py
