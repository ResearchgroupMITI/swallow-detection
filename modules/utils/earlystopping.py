"""Class to handle early stopping"""

class EarlyStopping:
    def __init__(self, hparams, idx):
        """Init function"""
        self.early_stopping = hparams.early_stopping
        self.early_stopping_mode = hparams.early_stopping_mode
        self.early_stopping_metric = hparams.early_stopping_metric
        self.patience = hparams.early_stopping_patience
        self.early_stopping_delta = hparams.early_stopping_delta
        self.early_stopping_counter = 0
        self.best_value = None
        self.index = idx

    def __call__(self, metric_dict):
        """Checks wehther run should stop early
        Returns:
            Whether to early stop the run
        """
        index_insert = self.early_stopping_metric.index("val_") + len("val_")
        new_string = self.early_stopping_metric[:index_insert] + f"fold{self.index}_" + self.early_stopping_metric[index_insert:]
        metric_val = metric_dict[new_string]
        if self.best_value is None:
            self.best_value = metric_val
        
        if self.early_stopping_mode == 'max':
            if metric_val > (self.best_value + self.early_stopping_delta):
                self.early_stopping_counter = 0
                self.best_value = metric_val
            else:
                self.early_stopping_counter +=1        
        elif self.early_stopping_mode == 'min':
            if metric_val < (self.best_value + self.early_stopping_delta):
                self.early_stopping_counter = 0
                self.best_value = metric_val
            else:
                self.early_stopping_counter +=1
        if self.early_stopping:
            if self.early_stopping_counter >= self.patience:
                return True
        return False




