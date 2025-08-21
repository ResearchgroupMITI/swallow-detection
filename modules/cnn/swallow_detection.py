import os
import torch
import torchmetrics
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm import tqdm
import wandb
import numpy as np
from torch_lr_finder import LRFinder
from modules.utils.modelsaving import ModelSaving
from modules.utils.dataloader import TestdatatLoader
from modules.utils.evaluationhelper import EvalHelper
from modules.utils.earlystopping import EarlyStopping
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class FeatureExtraction():
    """Feature extraction class"""
    def __init__(self, hparams, model, dataset, idx):
        """Initializes feature extraction class

        Args:
            hparams: hyperparameters
            model: model to be used for tr                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        aining/evaluation
            dataset: dataset
        """
        super(FeatureExtraction, self).__init__()
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.index = idx+1
        self.hparams = hparams
        self.model = model.to(self.device)
        self.dataset = dataset
        self.init_metrics()
        self.log_vars = nn.Parameter(torch.zeros(2))
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.trainloader, self.valloader = self.create_dataloaders()
        self.optimizer = self.configure_optimizers()
        self.early_stopping = EarlyStopping(hparams, self.index)
        self.modelsaving = ModelSaving(hparams, self.index)
        self.test_data = TestdatatLoader(hparams, self.dataset.val_splits[self.index-1])

        #learning rate finder
        #self.get_learning_rate()

        # store model
        self.pickle_path = None

        #Logging
        self.writer = SummaryWriter(log_dir=self.hparams.output_path)
        wandb.init(entity="miti", config=self.hparams, project=self.hparams.wandb_project_name, name=self.hparams.name, mode=self.hparams.wandb_mode, dir=self.hparams.output_path)

    def get_learning_rate(self):
        model = self.model
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
        lr_finder = LRFinder(model, optimizer, criterion, device=self.device)
        lr_finder.range_test(self.trainloader, end_lr=100, num_iter=100)
        lr_finder.plot()
        lr_finder.reset()

    def train(self):
        """Main training routine"""

        for epoch in tqdm(range(1, self.hparams.max_epochs + 1)):
            self.model.train()

            train_raw_values_list = []
            val_raw_values_list = []
            for train_batch in tqdm(self.trainloader, leave=False):
                train_p_values, train_y_values = self.training_step(train_batch)
                train_raw_values_list.append([train_p_values, train_y_values])
            self.model.eval()
            with torch.no_grad():
                for val_batch in tqdm(self.valloader, leave=False):
                    val_p_values, val_y_values = self.validation_step(val_batch)
                    val_raw_values_list.append([val_p_values, val_y_values])
            stop_early = self.end_of_epoch(train_raw_values_list, val_raw_values_list, epoch)
            if stop_early and (epoch >= self.hparams.min_epochs):
                break
        if self.hparams.test_model:
            self.test_step()

    def init_metrics(self):
        """Creates metrics"""
        self.metrics_names = ['precision', 'recall', 'specificity', 'accuracy', 'f1']
        classes_to_use = self.dataset.data['train'].class_names
        self.train_feature_metrics_names = ['train_' + metric_name + '_' + class_to_use for metric_name in self.metrics_names for class_to_use in classes_to_use]
        self.val_feature_metrics_names = ['val_' + metric_name + '_' + class_to_use for metric_name in self.metrics_names for class_to_use in classes_to_use]

        self.precision = torchmetrics.Precision(task="binary").to(self.device)
        self.recall = torchmetrics.Recall(task="binary").to(self.device)
        self.specificity = torchmetrics.Specificity(task="binary").to(self.device)
        self.acc = torchmetrics.Accuracy(task="binary").to(self.device)
        self.f1 = torchmetrics.F1Score(task="binary").to(self.device)

    def end_of_epoch(self, train_raw_values_list, val_raw_values_list, epoch):
        """Gets called at the end of epoch
        Calculates and logs metrics
        Saves model if best
        Possibly stops early

        Args:
            train_metrics_dicts: training metrics
            val_metrics_dicts: validation metrics
            epoch: current epoch

        Returns:
            stop_early (bool): whether to stop early
        """
        train_p_values_list = []
        train_y_values_list = []
        val_p_values_list = []
        val_y_values_list = []
        for sublist in train_raw_values_list:
            train_p_values_list.extend(sublist[0])
            train_y_values_list.extend(sublist[1])
        for sublist in val_raw_values_list:
            val_p_values_list.extend(sublist[0])
            val_y_values_list.extend(sublist[1])

        train_metric_dict = self.calc_metrics(torch.stack(train_p_values_list), torch.stack(train_y_values_list), 'train')
        val_metric_dict = self.calc_metrics(torch.stack(val_p_values_list), torch.stack(val_y_values_list), 'val')

        metric_dict = {**train_metric_dict, **val_metric_dict}

        wandb.log(metric_dict)

        self.modelsaving(metric_dict, self.model)
        stop_early = self.early_stopping(metric_dict)
        return stop_early

    def calc_metrics(self, p_values_tensor, y_values_tensor, mode):
        #Calculate metrics
        metric_dict = {}

        #Calculate loss
        metric_dict[f"{mode}_fold{self.index}_loss"] = self.bce_loss(p_values_tensor, y_values_tensor)

        #Sigmoid activation for correct calculation of other metrics
        p_values_tensor = torch.sigmoid(p_values_tensor)

        #Overall metrics
        metric_dict[f"{mode}_fold{self.index}_accuracy"] = self.acc(p_values_tensor, y_values_tensor)
        metric_dict[f"{mode}_fold{self.index}_f1"] = self.f1(p_values_tensor, y_values_tensor)
        metric_dict[f"{mode}_fold{self.index}_precision"] = self.precision(p_values_tensor, y_values_tensor.long())
        metric_dict[f"{mode}_fold{self.index}_recall"] = self.recall(p_values_tensor, y_values_tensor)
        metric_dict[f"{mode}_fold{self.index}_specificity"] = self.specificity(p_values_tensor, y_values_tensor)

        if mode == "train":
            feature_metrics_names = self.train_feature_metrics_names
        elif mode == "val":
            feature_metrics_names = self.val_feature_metrics_names

        for feature_metric_name in feature_metrics_names:
            single_feature_index = self.dataset.data['train'].class_name_to_num[feature_metric_name.split('_')[-1]]
            p_single_feature = p_values_tensor[:,single_feature_index]
            y_single_feature = y_values_tensor[:, single_feature_index]
            if (f"{mode}_accuracy_") in feature_metric_name:
                metric_dict[feature_metric_name] = self.acc(p_single_feature, y_single_feature)
            elif (f"{mode}_f1_") in feature_metric_name:
                metric_dict[feature_metric_name] = self.f1(p_single_feature, y_single_feature) 
            elif (f"{mode}_precision_") in feature_metric_name:
                metric_dict[feature_metric_name] = self.precision(p_single_feature, y_single_feature)
            elif (f"{mode}_recall_") in feature_metric_name:
                metric_dict[feature_metric_name] = self.recall(p_single_feature, y_single_feature)
            elif (f"{mode}_specificity_") in feature_metric_name:
                metric_dict[feature_metric_name] = self.specificity(p_single_feature, y_single_feature)
        return metric_dict

    # ---------------------
    # TRAINING
    # ---------------------

    def compute_loss(self, p_features, y_features):
        """Calculates loss

        Args:
            p_features: predicted features
            y_features: true features

        Returns:
            loss: loss value
        """
        #labels_features = torch.argmax(y_features, dim=1)
        loss = self.bce_loss(p_features, y_features)
        return loss

    def training_step(self, batch):
        """Does one training step

        Args:
            batch (list): Contains image tensor, ground truth phases and tools

        Returns:
            metric_dict: dict of training metrics
        """
        x, y_features, id = batch
        x, y_features = x.to(self.device), y_features.to(self.device)
        self.optimizer.zero_grad()
        p_features = self.model.forward(x)
        train_loss = self.compute_loss(p_features, y_features)
        train_loss.backward()
        self.optimizer.step()

        return p_features, y_features

    def validation_step(self, batch):
        """Does one validation step

        Args:
            batch (list): Contains image tensor, ground truth phases and tools

        Returns:
            metric_dict: dict of training metrics
        """
        x, y_features, id = batch
        x, y_features = x.to(self.device), y_features.to(self.device)

        p_features = self.model.forward(x)

        return p_features, y_features

    # ---------------------
    # TEST SETUP
    # ---------------------
    def get_smoothed_array(self, array) -> torch.Tensor:
        """to be defined"""
        window_size = 30
        array = array.rolling(window=window_size, center=True).mean()
        array = np.array(array.fillna(0))
        array = array.clip(-200, 300)
        array = (255*(array - np.min(array))/np.ptp(array)).astype(int)
        array = torch.Tensor(array).to(device=self.device)
        return array

    def get_resized_batch(self, windowI) -> torch.Tensor:
        """tbd"""
        resized_array = TF.resize(windowI, size=[224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=False)
        resized_array = torch.stack([resized_array] * 3, dim=1)
        return resized_array

    def test_step(self):
        """to be defined"""
        best_model_path = os.path.join(self.hparams.output_path, "models",
                                       f"best_model_fold{self.index}.pt")
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        # TODO parallelize and use data loader (see deploy_efficientnet.py)
        with torch.no_grad():
            print(f"Cross validation fold {self.index}")
            for k in range(0, len(self.test_data.data_indices)):
                pat_id = self.test_data.data_indices[k]
                manometry = self.get_smoothed_array(self.test_data.dataset[pat_id])
                # indices = []
                probability = []
                output = []
                print(f"Cross validation fold {self.index} patient {pat_id}")
                for i in tqdm(range(0, manometry.shape[0] - self.hparams.window_size, self.hparams.step_size * self.hparams.test_batch_size), total=manometry.shape[0] // (self.hparams.step_size *self.hparams.test_batch_size)):
                    # Create a batch of window tensors
                    batch_window_tensors = []
                    batch_indices = []
                    for j in range(self.hparams.test_batch_size):
                        index = i + j * self.hparams.step_size
                        if index + self.hparams.window_size > manometry.shape[0]:
                            break
                        # Slice the manometry tensor to get the window
                        manometry_window_i = manometry[index:index+self.hparams.window_size, :].T
                        # Append the transposed window to the list
                        batch_window_tensors.append(manometry_window_i)
                        batch_indices.append(index)
                        # Stack the batch of tensors along a new dimension
                    stacked_batch_tensor = torch.stack(batch_window_tensors, dim=0)
                    resized_batch = self.get_resized_batch(stacked_batch_tensor)
                    out = self.model.forward(resized_batch)
                    out = torch.sigmoid(out)
                    max_values, max_indices = torch.max(out, dim=1)

                    # append pre defined lists
                    probability.append(max_values)
                    output.append(max_indices)

                # stack and flatten output lists
                # check if stacking is possible --> otherwise padding is necessary
                if probability[-1].shape[0] != self.hparams.test_batch_size:
                    # probability padding for stacking
                    pad = self.hparams.test_batch_size - j
                    last_tensor_p = probability[-1].unsqueeze(0)
                    padded_last_tensor_p = F.pad(last_tensor_p, (0, pad), mode='constant', value=0).squeeze(0)
                    probability[-1] = padded_last_tensor_p
                    probability_stack = torch.stack(probability, dim=0)
                    probability_flatten = torch.flatten(probability_stack)[:-pad]
                    # output padding for stacking
                    last_tensor_o = output[-1].unsqueeze(0)
                    padded_last_tensor_o = F.pad(last_tensor_o, (0, pad), mode='constant', value=0).squeeze(0)
                    output[-1] = padded_last_tensor_o
                    output_stack = torch.stack(output, dim=0)
                    output_flatten = torch.flatten(output_stack)[:-pad]
                else:
                    probability_stack = torch.stack(probability, dim=0)
                    probability_flatten = torch.flatten(probability_stack)
                    output_stack = torch.stack(output, dim=0)
                    output_flatten = torch.flatten(output_stack)

                # multiplication of probability and ouput
                swallo_conf = probability_flatten*output_flatten

                # call evaluation helper class
                helper = EvalHelper(self.hparams, self.device, pat_id, swallo_conf, self.index)
                test_metric_dict = helper()
                wandb.log(test_metric_dict)
                print("\n")

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.SGD(self.model.parameters(),
                               lr=self.hparams.learning_rate,
                               weight_decay=self.hparams.weight_decay)
        return optimizer

    def create_dataloaders(self):
        """Creates dataloaders

        Returns:
            3 dataloaders for training, validation, testing
        """
        trainloader = DataLoader(
            dataset = self.dataset.data["train"],
            batch_size = self.hparams.batch_size,
            shuffle=True,
            num_workers = self.hparams.num_workers,
            pin_memory = False
        )
        valloader = DataLoader(
            dataset = self.dataset.data["val"],
            batch_size = self.hparams.batch_size,
            shuffle=True,
            num_workers = self.hparams.num_workers,
            pin_memory = False
        )

        return trainloader, valloader

    @staticmethod
    def add_module_specific_args(parser):
        """Adds module specific args"""
        swallow_module = parser.add_argument_group(
            title='swallow_module specific args options')

        # HPERPARAMTER
        swallow_module.add_argument("--learning_rate",
                                      default=0.0005,
                                      type=float)
        swallow_module.add_argument("--weight_decay",
                                      default=0.01,
                                      type=float)
        swallow_module.add_argument("--batch_size", default=32, type=int)
        swallow_module.add_argument("--test_model", action="store_true")

        # WEIGHTS and BIAS
        swallow_module.add_argument("--wandb_mode",
                                         default="online",
                                         choices=["online", "offline", "disabled"],
                                         type=str)
        swallow_module.add_argument("--wandb_project_name",
                                        default='trash',
                                        type=str)

        # EARLY STOPPING
        swallow_module.add_argument("--early_stopping",
                                         action="store_true")
        swallow_module.add_argument("--early_stopping_mode",
                                         default = "max",
                                         choices=["min", "max"],
                                         type=str)
        swallow_module.add_argument("--early_stopping_metric",
                                         default="val_loss",
                                         type=str)
        swallow_module.add_argument("--early_stopping_patience",
                                         default=5,
                                         type=int)
        swallow_module.add_argument("--early_stopping_delta",
                                         default=0.0,
                                         type=float)

        # SAVING
        swallow_module.add_argument("--save_model_metric",
                                         default="val_jaccardindex",
                                         type=str)
        swallow_module.add_argument("--save_model_mode",
                                         default="max",
                                         choices=["min", "max"],
                                         type=str)
        swallow_module.add_argument("--save_best_model",
                                         action="store_true")
        swallow_module.add_argument("--model_name",
                                         default="Please_define_model_name",
                                         type=str)

        # TESTING
        swallow_module.add_argument("--window_size",
                                         default=500,
                                         type=int)
        swallow_module.add_argument("--step_size",
                                         default=1,
                                         type=int)
        swallow_module.add_argument("--test_batch_size",
                                         default=128,
                                         type=int)
        swallow_module.add_argument("--threshold",
                                         default=0.2,
                                         type=float)
        swallow_module.add_argument("--rolling_window_size",
                                         default=500,
                                         type=int)
        swallow_module.add_argument("--min_distance",
                                         default=200,
                                         type=int)
        swallow_module.add_argument("--det_distance",
                                         default=400,
                                         type=int)
        swallow_module.add_argument("--min_length",
                                         default=2,
                                         type=int)
        swallow_module.add_argument("--save_output",
                                         action="store_true")

        return parser
