directory="/tmp/pretrained_models"

# Create the directory if it does not exist
if [ ! -d "$directory" ]; then
    mkdir -p "$directory"
fi


# copy mixtral
sudo chmod -R 777 /tmp/pretrained_models


### Since each node have different speed to download the model, we need to wait for all nodes to finish downloading the model before we start the training.
### Otherwise, may face the issue RuntimeError: Timed out initializing process group in store based barrier on rank: 2, for key: store_based_barrier_key:1 (world_size=32, worker_count=23, timeout=0:30:00)

# Start time
start_time=$(date +%s)

echo "Start downloading Mixtral-8x7B-Instruct-v0.1"

sudo azcopy copy "Mixtral-8x7B-Instruct-v0.1" "/tmp/pretrained_models"  --recursive=true --overwrite=false

# End time
end_time=$(date +%s)


# Calculate duration
duration=$((end_time - start_time))

# Desired minimum duration in seconds (70 minutes * 60 seconds/minute)
min_duration=$((70 * 60))

# If the duration is less than the minimum desired duration, wait for the difference
if [ $duration -lt $min_duration ]; then
    wait_time=$((min_duration - duration))
    echo "Waiting for additional $wait_time seconds to meet the 60-minute requirement."
    sleep $wait_time
else
    echo "No need to wait."
fi

echo "Download and wait complete."