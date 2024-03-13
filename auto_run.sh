
#!/bin/bash

# Path to the directory containing images_batch_* directories
batches_dir="/shared/anastasio-s2/SI/TCVAE/DL_feature_interpretation/dataset_with_padding/train"

# Loop through each images_batch_* directory
for batch_dir in ${batches_dir}/images_batch_*; do
    # Determine the batch number to correctly specify the output directory
    batch_num=$(basename $batch_dir)
    
    # Specify the output directory
    output_dir="/shared/anastasio-s2/SI/TCVAE/DL_feature_interpretation/result/${batch_num}"

    echo "Starting processing for ${batch_num} at $(date)"
    # Call run_tile.sh with the current batch directory and corresponding output directory
    
    ./run_tile.sh "${batch_dir}" "${output_dir}"
    echo "Finished processing for ${batch_num} at $(date)"
done
