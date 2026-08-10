from scipy import sparse
import numpy as np
import torch
import time

class IoU(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self._total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)

    def update(self, y_pred, y_true):
        """

        Args:
            y_pred: 1-D
            y_true: 1-D

        Returns:

        """
        v = np.ones_like(y_pred)
        cm = sparse.coo_matrix((v, (y_true, y_pred)), shape=(self.num_classes, self.num_classes), dtype=np.float32)
        self._total += cm

    def cm_value(self):
        dense_cm = self._total.toarray()
        row_sum = np.sum(dense_cm, axis=1)
        dense_cm /= row_sum[:, None]
        return dense_cm

    def value(self):
        dense_cm = self._total.toarray()
        # print(dense_cm)
        # np.save('cm',dense_cm)
        return compute_iou_per_class(dense_cm)


def compute_iou_per_class(confusion_matrix):
    """
    Args:
        confusion_matrix: numpy array [num_classes, num_classes] row - gt, col - pred
    Returns:
        iou_per_class: float32 [num_classes, ]
    """
    sum_over_row = np.sum(confusion_matrix, axis=0)
    sum_over_col = np.sum(confusion_matrix, axis=1)
    diag = np.diag(confusion_matrix)
    denominator = sum_over_row + sum_over_col - diag

    iou_per_class = diag / denominator

    return iou_per_class



if __name__ == '__main__':
    iou = IoU(3)
    y_true = np.array([0, 0, 1, 2, 2, 1, 1, 0])
    y_pred = np.array([0, 1, 1 ,2, 1, 0, 2, 1])
    iou.update(y_pred, y_true)
    print(iou.cm_value())