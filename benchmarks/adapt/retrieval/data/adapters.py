from ..utils.file_utils import load_samples
from collections import defaultdict
from ..utils.logger import get_logger
from pathlib import Path


logger = get_logger()


class FoodiML:

    def __init__(self, data_path, data_split):

        data_split = data_split.replace('dev', 'val')
        self.data_split = data_split
        self.data_path = Path(data_path)
        self.samples_path = (
            self.data_path / 'samples'  # TODO: change to 'samples'
        )
        self.data = load_samples(self.samples_path, self.data_split)
        self.image_ids, self.img_dict, self.img_captions = self._get_img_ids()

        logger.info(f'[FoodiML] Loaded {len(self.img_captions)} images annotated ')

    def _get_img_ids(self):
        image_ids = list(self.data["img_id"].values)
        img_dict = {}
        annotations = defaultdict(list)
        for _, row in self.data.iterrows():
            img_dict[row["img_id"]] = row["s3_path"]
            annotations[row["img_id"]].extend([row["caption"]])

        return image_ids, img_dict, annotations

    def get_image_id_by_filename(self, filename):
        return self.img_dict[filename]['imgid']

    def get_captions_by_image_id(self, img_id):
        return self.img_captions[img_id]

    def get_filename_by_image_id(self, image_id):
        return self.img_dict[image_id]

    def __call__(self, filename):
        return self.img_dict[filename]

    def __len__(self, ):
        return len(self.img_captions)


