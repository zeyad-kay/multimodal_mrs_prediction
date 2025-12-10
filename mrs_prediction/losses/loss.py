import torch

class MultiTaskLoss(torch.nn.Module):
    def __init__(self, losses, tasks):
        super().__init__()
        self._loss_dict = {
            "mean_squared_error": torch.nn.MSELoss,
            "binary_cross_entropy": torch.nn.BCEWithLogitsLoss
        }
        self.losses = [self._loss_dict[loss]() for loss in losses]
        self.tasks = tasks

    def forward(self, outputs, labels, weights):
        task_loss = []
        total_loss = torch.tensor(0.0, device=outputs.device)
        for i, w_loss in enumerate(zip(weights, self.losses)):
            weight, loss_fn = w_loss
            loss = loss_fn(outputs[:, i], labels[:, i])
            total_loss += (weight * loss)
            task_loss.append((self.tasks[i], loss.item()))
        return task_loss, total_loss
