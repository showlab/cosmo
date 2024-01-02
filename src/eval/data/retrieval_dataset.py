import os
from torch.utils.data import Dataset
import json
from PIL import Image

class RetrievalDataset(Dataset):
    def __init__(
        self,
        image_dir_path,
        annotations_path,
        is_train,
        dataset_name,
    ):
        self.image_dir_path = image_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name
        print("Loading retrieval from", annotations_path)
        full_annotations = json.load(open(annotations_path))

        for i in range(len(full_annotations)):
            self.annotations.append(full_annotations[i])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image = Image.open(
                os.path.join(
                    self.image_dir_path, self.annotations[idx]["image"]
                )
            )
        elif self.dataset_name == "flickr":
            image = Image.open(
                os.path.join(
                    self.image_dir_path, self.annotations[idx]["image"]
                )
            )
        image.load()
        caption = self.annotations[idx]["caption"][0]
        return {
            "image": image,
            "caption": caption,
            "image_id": idx
        }
