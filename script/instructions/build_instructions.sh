#! /usr/bin/bash
# Usage: `bash script/instructions/build_instructions.sh`

# Main script to build dataset.

out_root=$PROJECT_ROOT/data/instructions
hf_root=$PROJECT_ROOT/data/instructions_hf
mkdir -p $out_root
mkdir -p $hf_root

# Validate schema and download all datasets.
python script/instructions/validate.py   # Validate schema.

# Create versions with different context windows.
for context_window in 4096 8192 16384
do
    out_dir=$out_root/${context_window}
    python script/instructions/instantiate.py \
        --templates 0 \
        --n_instances 2500 \
        --context_window $context_window \
        --workers 60 \
        --out_dir $out_dir

    # Write-protect the data just to be safe.
    chmod -R a-w $out_dir/*
done

# Some quick checks that everything looks ok.
pytest script/instructions/test_instances.py

# Merge instances from all tasks into a single file for HF upload.

for context_window in 4096 8192 16384
do
    inst_dir=$out_root/${context_window}
    hf_dir=$hf_root/${context_window}
    python script/instructions/combine.py \
        --in_dir $inst_dir \
        --out_dir $hf_dir

    # Write-protect the data just to be safe.
    chmod -R a-w $hf_dir/*
done
