"""
since webdataset can not count sample directly, we write this script to count the number of samples in each tar file
example:

python src/utils/count_webdataset_sample.py --data_dir /home/linjie/blob/vigstandard_data/v-jinpewang/dataset/webvid2_5m/train --num_extra_anno 2

python src/utils/count_webdataset_sample.py --data_dir /home/linjie/blob/vigstandard_data/v-jinpewang/dataset/cc12m_wds/images --num_extra_anno 2

python src/utils/count_webdataset_sample.py --data_dir /home/linjie/blob/vigstandard_data/v-jinpewang/dataset/laion400m_wds_w_gen_caption --num_extra_anno 2

python src/utils/count_webdataset_sample.py --data_dir /home/linjie/blob/vigstandard_data/v-jinpewang/dataset/data_comp_1b/shards --num_extra_anno 2

"""
import os
import tarfile
import json
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Count the number of samples in each tar file in a directory.')
parser.add_argument('--data_dir', type=str, help='The directory that contains your tar files', default="/datadrive_d/jinpeng/Code/videogpt4/data/interim/image_text_pairs/cc3m_webdataset")
parser.add_argument('--num_extra_anno', type=int, default=0, help='The number of extra anno for each sample, for example, if sample is 1.jpg, 1.txt, 1.jpg, then should be 2')
args = parser.parse_args()

# Get data_dir from arguments
data_dir = args.data_dir

# Create a dict to store the number of samples for each tar file
num_samples_dict = {}

count = 0
# Iterate over each tar file
for filename in os.listdir(data_dir):
    if filename.endswith(".tar"):
        try:
            with tarfile.open(os.path.join(data_dir, filename), "r") as tar:
                # Count the number of members in the tar file
                try:
                    num_samples = len(tar.getmembers())
                    # The number of samples is the number of members minus 2
                    num_samples_dict[filename] = num_samples//(1+args.num_extra_anno)
                except tarfile.ReadError as e:
                    print(f"Error reading {filename}: {e}")
                    break
        except Exception as e:
            print(f"Error opening {filename}: {e}")
            continue
        try:
            print(f"Finished counting samples in {filename} with {num_samples_dict[filename]} samples")
        except Exception as e:
            print(f"Error counting samples in {filename}: {e}")
            continue
        count += 1
        if count % 10 == 0:
            print(f"Finished counting samples in {count} tar files")

# Save the number of samples to a JSON file
with open(os.path.join(data_dir, "num_samples.json"), "w") as f:
    json.dump(num_samples_dict, f)