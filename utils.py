import numpy as np
import torch

class GrooundTruthScheduler:
    def __init__(self,initial_groundtruth=1.0,start_epoch=0, patient_epochs=30, desend_step=0.01, cooldown=10,
                 net_loss_weight=1,cell_loss_weight=1,at_loss_weight=1):
        self.net_loss = None
        self.cell_loss = None
        self.at_loss = None
        self.current_epoch = start_epoch
        self.best_epoch = None
        self.min_loss = None
        self.patient_epochs = patient_epochs
        self.desend_step = desend_step
        self.cooldown = cooldown
        self.net_loss_weight = net_loss_weight
        self.cell_loss_weight = cell_loss_weight
        self.at_loss_weight = at_loss_weight
        self.current_groundtruth = initial_groundtruth

    def step(self, net_loss, cell_loss, at_loss, epoch):
        self.net_loss = net_loss
        self.cell_loss = cell_loss
        self.at_loss = at_loss
        self.current_epoch = epoch
        loss = self.loss()
        if self.min_loss is None:
            self.min_loss = loss
            self.best_epoch = self.current_epoch
            self.current_groundtruth -= self.desend_step
        else:
            if loss < 5*self.min_loss:
                self.min_loss = loss
                last_best_epoch = self.best_epoch
                self.best_epoch = self.current_epoch
                if self.current_epoch - last_best_epoch > self.cooldown:
                    self.current_groundtruth -= self.desend_step
                    self.current_groundtruth = max(0.0, self.current_groundtruth)
            else:
                if self.current_epoch - self.best_epoch > self.patient_epochs:
                    self.current_groundtruth += self.desend_step
                    self.current_groundtruth = min(1.0, self.current_groundtruth)
        return self.current_groundtruth

    def loss(self):
        return self.net_loss*self.net_loss_weight+self.cell_loss*self.cell_loss_weight+self.at_loss*self.at_loss_weight


class AdaptiveLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, mode='min', factor=0.5, increase_factor=1.1,
                 patience=10, threshold=0.05, cooldown=0, warmup_epochs=30,
                 min_lr=1e-7, max_lr=1e-2, noise_pct=0.67, noise_std=1.0,
                 last_epoch=-1):
        """
        自适应学习率调度器：
        - `metric` 在 `threshold` 内震荡超过 `patience` 轮时降低学习率。
        - `metric` 明显劣化并超过 `patience` 轮时增加学习率（受 `max_lr` 限制）。
        - 支持 `warmup_epochs` 进行预热。
        - `min_lr` 和 `max_lr` 约束学习率范围。
        - `noise_pct` 控制噪声扰动幅度（相对）。
        - `noise_std` 控制噪声标准差（绝对）。

        :param optimizer: 优化器
        :param mode: 'min'（metric 越小越好）或 'max'（metric 越大越好）
        :param factor: 降低学习率的因子（默认 0.5）
        :param increase_factor: 增加学习率的因子（默认 1.1）
        :param patience: 忍耐次数，超过后调整学习率
        :param threshold: 震荡检测的最小变化比率
        :param cooldown: 学习率调整后冷却轮数
        :param warmup_epochs: 预热轮数
        :param min_lr: 最小学习率
        :param max_lr: 最大学习率
        :param noise_pct: 学习率的随机噪声（相对比例，如 0.1 表示 10% 变化）
        :param noise_std: 额外高斯噪声的标准差（绝对值）
        """
        self.mode = mode
        self.factor = factor
        self.increase_factor = increase_factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.noise_pct = noise_pct
        self.noise_std = noise_std

        self.best_metric = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.warmup_counter = 0

        super().__init__(optimizer, last_epoch)

    def step(self, metric=None):
        """
        根据 metric 调整学习率
        """
        if metric is None:
            return

        # 处理 warm-up 阶段
        if self.warmup_counter < self.warmup_epochs:
            self.warmup_counter += 1
            for i, base_lr in enumerate(self.base_lrs):
                new_lr = base_lr * (self.warmup_counter / self.warmup_epochs)
                self.optimizer.param_groups[i]['lr'] = new_lr
            return

        # 计算 metric 变化趋势
        if self.best_metric is None:
            self.best_metric = metric
            return

        if (self.mode == 'min' and metric < self.best_metric * (1 - self.threshold)) or \
                (self.mode == 'max' and metric > self.best_metric * (1 + self.threshold)):
            self.best_metric = metric
            self.num_bad_epochs = 0  # 重置计数
        else:
            self.num_bad_epochs += 1

        # 冷却期处理
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        # 判断是否要调整学习率
        if self.num_bad_epochs > self.patience:
            self._adjust_learning_rate(metric)
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown  # 进入 cooldown 期

    def _adjust_learning_rate(self, metric):
        """
        调整学习率
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']

            if (self.mode == 'min' and metric > self.best_metric * (1 + self.threshold)) or \
                    (self.mode == 'max' and metric < self.best_metric * (1 - self.threshold)):
                # 训练效果变差，增加学习率
                new_lr = min(old_lr * self.increase_factor, self.max_lr)
                self.best_metric = metric
            else:
                # 训练震荡，降低学习率
                new_lr = max(old_lr * self.factor, self.min_lr)

            # 添加噪声扰动
            new_lr = self._apply_noise(new_lr)

            param_group['lr'] = new_lr

    def _apply_noise(self, lr):
        """
        根据 `noise_pct` 和 `noise_std` 添加噪声
        """
        if self.noise_pct > 0 and self.noise_std > 0:
            noise_factor = np.random.uniform(1.0 - self.noise_pct, 1.0 + self.noise_pct)
            lr += lr * noise_factor * np.random.normal(0, self.noise_std)

        return max(self.min_lr, min(lr, self.max_lr))  # 确保学习率在范围内

    def get_last_lr(self):
        """
        获取当前学习率
        """
        return [group['lr'] for group in self.optimizer.param_groups]


def log_cosh_loss(y_pred, y_true):
    return torch.mean(torch.log(torch.cosh(y_pred - y_true)))


def mae_loss(y_pred, y_ture):
    return torch.mean(torch.abs(y_pred-y_ture))
