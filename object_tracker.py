import numpy as np

from scipy.spatial import distance as dist
from collections import OrderedDict


class ObjectTracker():

    def __init__(self, max_frames_disappeared=50):

        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_frames_disappeared = max_frames_disappeared

    def register(self, centroid):

        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def unregister(self, object_id):

        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):

        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_frames_disappeared:
                    self.unregister(object_id)
            
            return self.objects
        else:
            input_centroids = np.zeros((len(rects), 2), dtype="int")

            for (i, (x1, y1, x2, y2)) in enumerate(rects):
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                input_centroids[i] = (cx, cy)

            if len(self.objects) == 0:
                for i in range(0, len(input_centroids)):
                    self.register(input_centroids[i])
            else:
                object_ids = list(self.objects.keys())
                object_centroids = list(self.objects.values())

                distance = dist.cdist(np.array(object_centroids), input_centroids)
                rows = distance.min(axis=1).argsort()
                cols = distance.argmin(axis=1)[rows]

                used_rows = set()
                used_cols = set()

                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue

                    object_id = object_ids[row]
                    self.objects[object_id] = input_centroids[col]
                    self.disappeared[object_id] = 0

                    used_rows.add(row)
                    used_cols.add(col)

                    unused_rows = set(range(0, distance.shape[0])).difference(used_rows)
                    unused_cols = set(range(0, distance.shape[1])).difference(used_cols)

                    if distance.shape[0] >= distance.shape[1]:
                        for row in unused_rows:
                            object_id = object_ids[row]
                            self.disappeared[object_id] += 1

                            if self.disappeared[object_id] > self.max_frames_disappeared:
                                self.unregister(object_id)
                    else:
                        for col in unused_cols:
                            self.register(input_centroids[col])

            return self.objects
