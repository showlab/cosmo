from torchvision import transforms

def image_augment(mode='train'):
    """
    # for eval/test
    # way1: resize and crop (drop performance 3% on coco captioning)
    #       transforms.Resize(256),
    #       transforms.CenterCrop(224),
    # way2: resize full image
    #       transforms.Resize((224, 224)),
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if mode == 'train':
        preproc = transforms.Compose([
            transforms.Lambda(lambda image: image.convert("RGB")), # !not sure if will slow down
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        preproc = transforms.Compose([
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    return preproc