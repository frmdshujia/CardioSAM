from __future__ import annotations

import random
from typing import Optional

from sam3.train.data.sam3_image_dataset import Datapoint


class RandomDropInputBbox:
    """
    Randomly drop input boxes from find queries.
    """

    def __init__(self, prob: float = 0.5):
        assert 0.0 <= prob <= 1.0
        self.prob = prob

    def __call__(self, datapoint: Datapoint, **kwargs):
        if random.random() > self.prob:
            return datapoint
        for query in datapoint.find_queries:
            query.input_bbox = None
        return datapoint

