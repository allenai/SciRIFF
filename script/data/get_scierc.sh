# Process the scierc dataset.
# Run this script from the `tasks` directory by invoking `bash script/get_scierc.sh`

# Exit if the dataset's already there.
if [ -e data/processed/scierc ]
then
    echo "Processed scierc dataset already found; exiting."
    exit 0
fi

current_dir=$(pwd)
wget -P data/preprocessing/downloads http://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz
cd data/preprocessing/downloads
tar -xvf sciERC_processed.tar.gz
rm -r processed_data/elmo
rm sciERC_processed.tar.gz

mv processed_data/json scierc
rmdir processed_data
cd $current_dir
mkdir -p data/preprocessing/processed/scierc
python script/data/process_scierc.py
