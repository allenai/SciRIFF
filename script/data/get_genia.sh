# Process the genia ataset.
# Run this script from the `tasks` directory by invoking `bash script/get_genia.sh`

# Exit if the dataset's already there.
if [ -e data/preprocessing/processed/genia ]
then
    echo "Processed genia dataset already found; exiting."
    exit 0
fi

tar -xvf data/lfs_data/genia.tar.gz -C data/preprocessing/downloads

mkdir -p data/preprocessing/processed/genia
python script/data/process_genia.py
