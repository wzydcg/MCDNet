"""
@Time ： 2021/7/1 13:33
@Auth ： wangbooming
@File ：Hausdorff_2D.py
@IDE ：PyCharm
"""
import numpy as np
from hausdorff import hausdorff_distance

# # two random 2D arrays (second dimension must match)
# np.random.seed(0)
# X = np.random.random((1000, 100))
# Y = np.random.random((5000, 100))
#
# # Test computation of Hausdorff distance with different base distances
# print("Hausdorff distance test: {0}".format(hausdorff_distance(X, Y, distance="manhattan")))
# print("Hausdorff distance test: {0}".format(hausdorff_distance(X, Y, distance="euclidean")))
# print("Hausdorff distance test: {0}".format(hausdorff_distance(X, Y, distance="chebyshev")))
# print("Hausdorff distance test: {0}".format(hausdorff_distance(X, Y, distance="cosine")))
#
#
# # For haversine, use 2D lat, lng coordinates
# def rand_lat_lng(N):
#     lats = np.random.uniform(-90, 90, N)
#     lngs = np.random.uniform(-180, 180, N)
#     return np.stack([lats, lngs], axis=-1)
#
#
# X = rand_lat_lng(100)
# Y = rand_lat_lng(250)
# print("Hausdorff haversine test: {0}".format(hausdorff_distance(X, Y, distance="haversine")))

class HausdorffAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = HausdorffAverage.get_hausdorff(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_hausdorff(logits, targets):
        hausdorff = []
        for class_index in range(targets.size()[1]):
            # inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            # union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            X = logits[:, class_index, :, :, :]
            Y = targets[:, class_index, :, :, :]
            hausdorff = hausdorff_distance(X, Y, distance="euclidean")
            hausdorff.append(hausdorff.item())
        return np.asarray(hausdorff)


train_dice = HausdorffAverage(3)

