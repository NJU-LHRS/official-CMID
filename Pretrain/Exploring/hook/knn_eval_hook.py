import torch
import logging
from tqdm import tqdm
from typing import Tuple
from .hookbase import HookBase
import torch.nn.functional as F
from torchmetrics.metric import Metric

logger = logging.getLogger("train")


class KnnEvaluate(HookBase):
    def __init__(self,
                 k: int = 20,
                 T: float = 0.07,
                 distance_fx: str = "cosine"):
        super(KnnEvaluate, self).__init__()
        self.knn = WeightedKNNClassifier(k=k, distance_fx=distance_fx, T=T)

    def before_epoch(self) -> None:
        self.trainer.model_or_module.eval()
        index = 0
        with tqdm(self.trainer.data_loader, unit="Batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {self.trainer.epoch}\tknn-Training: ")
                img = data["img"]
                output = self.trainer.model_or_module.eval_forward(img, self.trainer.device)
                self.knn.update(
                    train_features=output.detach(),
                    train_targets=target.to(output.device).detach(),
                )
                index += 1
                if index == 10:
                    break

        index = 0
        with tqdm(self.trainer.eval_loader, unit="Batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {self.trainer.epoch}\tknn-Testing: ")
                output = self.trainer.model_or_module.eval_forward(data, self.trainer.device)
                self.knn.update(test_features=output.detach(), test_targets=target.to(output.device).detach())
                index += 1
                if index == 10:
                    break

        val_knn_acc1, val_knn_acc5 = self.knn.compute()
        logger.info("Epoch: {0}\tTOP1: {1:.4f}\tTOP5: {2:.4f}".format(
            self.trainer.epoch, val_knn_acc1, val_knn_acc5
        ))
        self.trainer.log(self.trainer.epoch, val_knn_acc1=val_knn_acc1)
        self.trainer.log(self.trainer.epoch, val_knn_acc5=val_knn_acc5)


class WeightedKNNClassifier(Metric):
    def __init__(
            self,
            k: int = 20,
            T: float = 0.07,
            max_distance_matrix_size: int = int(5e6),
            distance_fx: str = "cosine",
            epsilon: float = 0.00001,
            dist_sync_on_step: bool = False,
    ):
        """Implements the weighted k-NN classifier used for evaluation.

        Args:
            k (int, optional): number of neighbors. Defaults to 20.
            T (float, optional): temperature for the exponential. Only used with cosine
                distance. Defaults to 0.07.
            max_distance_matrix_size (int, optional): maximum number of elements in the
                distance matrix. Defaults to 5e6.
            distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
                "euclidean". Defaults to "cosine".
            epsilon (float, optional): Small value for numerical stability. Only used with
                euclidean distance. Defaults to 0.00001.
            dist_sync_on_step (bool, optional): whether to sync distributed values at every
                step. Defaults to False.
        """

        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.k = k
        self.T = T
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon

        self.add_state("train_features", default=[], persistent=False)
        self.add_state("train_targets", default=[], persistent=False)
        self.add_state("test_features", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)

    def update(
            self,
            train_features: torch.Tensor = None,
            train_targets: torch.Tensor = None,
            test_features: torch.Tensor = None,
            test_targets: torch.Tensor = None,
    ):
        """Updates the memory banks. If train (test) features are passed as input, the
        corresponding train (test) targets must be passed as well.

        Args:
            train_features (torch.Tensor, optional): a batch of train features. Defaults to None.
            train_targets (torch.Tensor, optional): a batch of train targets. Defaults to None.
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    @torch.no_grad()
    def compute(self):
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """

        train_features = torch.cat(self.train_features)
        train_targets = torch.cat(self.train_targets)
        test_features = torch.cat(self.test_features)
        test_targets = torch.cat(self.test_targets)

        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_test_images = test_targets.size(0)
        num_train_images = train_targets.size(0)
        chunk_size = min(
            max(1, self.max_distance_matrix_size // num_train_images),
            num_test_images,
        )
        k = min(self.k, num_train_images)

        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in range(0, num_test_images, chunk_size):
            # get the features for test images
            features = test_features[idx: min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx: min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                similarities = torch.mm(features, train_features.t())
            elif self.distance_fx == "euclidean":
                similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
            else:
                raise NotImplementedError

            similarities, indices = similarities.topk(k, largest=True, sorted=True)
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                similarities = similarities.clone().div_(self.T).exp_()

            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    similarities.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = (
                    top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
            )  # top5 does not make sense if k < 5
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total

        self.reset()

        return top1, top5