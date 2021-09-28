from torchvision import transforms


def get_transform(
        split,
        resize_to=256,
        crop_size=224,
    ):

    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == 'train':
        t_list = [
            transforms.Resize(resize_to),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(crop_size),
        ]
    else:
        t_list = [
            transforms.Resize(resize_to),
            transforms.CenterCrop(crop_size),
        ]

    t_list.extend([transforms.ToTensor(), normalizer])
    transform = transforms.Compose(t_list)

    return transform

