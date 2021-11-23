from collections import defaultdict
import numpy as np
import pandas as pd

def compute_valid_answers(data: pd.DataFrame):
    """Generates the valid answers taking into account mutiple images and captions.
    For the following dataset we will create the dictionary with valid_answers:
    id    caption    hash    |    valid_answers
    0     ABC        X       |    0,1,2,4
    1     EFG        X       |    0,1,4
    2     ABC        Y       |    0,2
    3     HIJ        Z       |    3,
    4     KLM        X       |    0,1,4
    """
    valid_answers = {}
    print("Computing valid answers: ")
    for i in range(data.shape[0]):
        idxs_where_duplication = (data["caption"] == data["caption"].iloc[i]) | (
            data["s3_path"] == data["s3_path"].iloc[i]
        )
        list_indexes_duplication = list(
            np.where(np.array(idxs_where_duplication.to_list()) == True)[0]
        )
        valid_answers[data["img_id"].iloc[i]] = list_indexes_duplication
    return valid_answers


class adapter:
    def __init__(self, dataframe):
        self.data = dataframe

    def get_adapter(self):
        """For a given dataframe this adapter gives the image_id given an index of the dataframe
        for consistency the adapter will be a dictionary with a key image_ids, that will be a list
        that returns at index i the image id corresponding to it
        """
        image_ids = list(self.data["img_id"].values)
        img_dict = {}
        annotations = defaultdict(list)
        for _, row in self.data.iterrows():
            img_dict[row["img_id"]] = row["s3_path"]
            annotations[row["img_id"]].extend([row["caption"]])
        self.image_ids = image_ids
        self.img_dict = img_dict
        self.annotations = annotations
        return self