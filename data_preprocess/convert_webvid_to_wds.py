import argparse
import os
import pandas as pd
import webdataset as wds
import cv2
import sys

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--output_dir", type=str)
arg_parser.add_argument(
    "--annotations",
    type=str,
    help=".csv file containing annotations",
)
arg_parser.add_argument(
    "--data_dir",
    type=str,
    help="Path to the directory containing the videos",
)
args = arg_parser.parse_args()


def test_video(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Cannot open video: {video_path}")
        return False
    video.release()
    return True


def main():
    data_dir = args.data_dir
    annotations = pd.read_csv(args.annotations)
    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    wrong_count = 0
    with wds.ShardWriter(args.output_dir + "/%09d.tar", maxcount=5000) as sink:
        for index, row in annotations.iterrows():
            if index%100 == 0:
                print(f"{index} in total")
                print(f"Total number of wrong videos: {wrong_count}")
                print(f"Total number of videos: {count}")
            video_path = os.path.join(data_dir, str(row['videoid']) + '.mp4')
            # Create a json file
            json_annotation_for_one_video = row.to_dict()
            caption = row['name']
            if not os.path.exists(video_path):
                continue
            # Add the video to the tar file
            if test_video(video_path):
                with open(video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                sink.write({"__key__": "sample%06d" % count, "json": json_annotation_for_one_video, "mp4": video_bytes, "txt": caption})
                count += 1
            else:
                wrong_count += 1

if __name__ == "__main__":
    main()