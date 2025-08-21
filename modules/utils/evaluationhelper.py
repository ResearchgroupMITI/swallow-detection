"""to be defined"""

import os 
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle


class EvalHelper():
    def __init__(self, hparams, device, pat_id, prediction, index) -> None:
        self.device = device
        self.prediction = prediction
        self.pat_id = pat_id
        self.rolling_window_size = hparams.rolling_window_size #20
        self.threshold = hparams.threshold #0.2
        self.min_distance = hparams.min_distance #200
        self.det_distance = hparams.det_distance #400
        self.min_length = hparams.min_length #2
        self.save_output = hparams.save_output #yes
        self.model_name = hparams.model_name
        self.index = index
        self.data_root = hparams.data_root

    def __call__(self):
        swallow_conf_smoothed = self.rolling_window(self.prediction)
        swallow_conf_certain = self.threshold_to_binary(swallow_conf_smoothed)
        found_swallows = self.find_groups_of_ones(swallow_conf_certain)
        found_swallows = self.filter_groups_by_length(found_swallows)
        gt_swallows = self.load_pat_ground_truth()
        tdict = self.test_rec_prec(found_swallows, gt_swallows)

        return tdict

    def load_pat_ground_truth(self):
        """to be defined"""
        # TODO Pfad relativ setzen
        swallow_file_path = f"/media/mitiadmin/Micron_7450_1/miccai24/data/manometry_swallows_{self.pat_id}.csv"
        swallow_file = pd.read_csv(swallow_file_path, sep=",")
        gt_tensor = torch.tensor(swallow_file.values)
        return gt_tensor.to(self.device)

    def rolling_window(self, tensor):
        """running convolution filter"""
        conv_filter = torch.ones(self.rolling_window_size) / self.rolling_window_size
        conv_filter = conv_filter.to(self.device)
        tensor_smoothed = F.conv1d(tensor.unsqueeze(0).unsqueeze(0),
                                   conv_filter.unsqueeze(0).unsqueeze(0),
                                   padding=self.rolling_window_size//2)
        return tensor_smoothed.squeeze()[:-1]

    def threshold_to_binary(self, tensor):
        """to be defined"""
        return (tensor >= self.threshold).to(torch.int)

    def find_groups_of_ones(self, tensor):
        """to be defined"""
        # Convert tensor to include boundaries for ease of group identification
        ten = torch.tensor([0]).to(self.device)
        padded = torch.cat([ten, tensor, ten])
        # Find changes in state
        diff = torch.diff(padded)
        # Start and end indices of ones
        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0] - 1
        groups = torch.stack([starts, ends], dim=1)

        # Merge groups if they are closer than self.min_distance
        merged_groups = []
        merged_groups.append(groups[0])
        for group in groups[1:]:
            if group[0] - merged_groups[-1][1] - 1 <= self.min_distance:
                # Merge current group with the last one in the list
                merged_groups[-1] = torch.tensor([merged_groups[-1][0], group[1]])
            else:
                merged_groups.append(group)

        merged_groups = self.move_and_stack_tensors(merged_groups)

        return merged_groups

    def move_and_stack_tensors(self, tensor_list):
        moved_tensors = [tensor.to(self.device) for tensor in tensor_list]
        stacked_tensors = torch.stack(moved_tensors)
        return stacked_tensors

    def filter_groups_by_length(self, groups):
        """to be defined"""
        # and each row is a [start, end] pair.
        lengths = groups[:, 1] - groups[:, 0]
        # Find the indices where the length condition is satisfied
        filtered_indices = lengths > self.min_length
        # Select only those groups
        filtered_groups = groups[filtered_indices]
        return filtered_groups

    def write_pickle(self, det_start, true_start, output):
        """to bed defined"""
        data = {'detected_swallow_starts': det_start, 
                'true_swallow_starts': true_start,
                'output': output}

        # create folder
        out_folder = os.path.join(self.data_root, f'output_{self.model_name}')
        os.makedirs(out_folder, exist_ok=True)

        # Specify the file path for the pickle file
        pickle_file_path = f'{out_folder}/output_fold{self.index}_patient{self.pat_id}.pkl'

        # Save the dictionary containing the lists to a pickle file
        with open(pickle_file_path, 'wb') as f:
            pickle.dump(data, f)

    def write_txt(self, precision, recall, f1, mean_dist):
        """to be defined"""
        # create folder 
        metric_folder = os.path.join(self.data_root, f'output_{self.model_name}')
        os.makedirs(metric_folder, exist_ok=True)

        metric_file = os.path.join(metric_folder, f"metrics_fold{self.index}" + '.txt')
        with open(metric_file, 'a') as file:
            line = f"{self.pat_id} {precision} {recall} {f1} {mean_dist}\n"
            file.write(line)

    def test_rec_prec(self, found_swallows, gt_swallows):
        """to be defined"""
        correct_list = []
        dist_list = []
        ind_list = []
        likely_start_list = []
        print('Comparing detected swallows to ground truth...')
        for row in gt_swallows:
            correct = 0
            for found_swallow in found_swallows:
                likely_start = found_swallow[0] + self.prediction[found_swallow[0]:found_swallow[1]].argmax()
                if likely_start.item() in range(row[0] - int(self.det_distance / 2), row[0] + int(self.det_distance / 2)):
                    correct = 1
                    dist = likely_start-row[0]
                    correct_list.append(correct)
                    ind_list.append(correct)
                    dist_list.append(dist.item())
                    likely_start_list.append(likely_start.item())
            if not correct:
                ind_list.append(correct)
                likely_start_list.append(-1)

        recall = len(correct_list)/len(gt_swallows)
        precision = len(correct_list)/len(found_swallows)
        f1 = (2 * recall * precision)/(recall + precision)
        # mean_dist = torch.mean(torch.stack(dist_list).to(torch.float)).cpu().item()
        mean_dist = np.mean(dist_list)

        print("---------------",'\n',
              f'Total swallows: {len(gt_swallows)}','\n',
              f'Found swallows: {len(found_swallows)}','\n',
              f'Correct swallows: {len(correct_list)}','\n',
              f'Mean distance of detected swallows: {np.round(mean_dist, decimals=3)}','\n',
              f'Precision: {np.round(precision, decimals=5)}','\n',
              f'Recall: {np.round(recall, decimals=5)}','\n',
              f'F1 score: {np.round(f1, decimals=5)}','\n',
              "---------------")

        if self.save_output:
            self.write_pickle(likely_start_list ,gt_swallows[:,0].cpu().tolist(), ind_list)
            self.write_txt(precision, recall, f1, mean_dist)

        tdict = {}
        tdict["test_precision"] = precision
        tdict["test_recall"] = recall
        tdict["test_f1"] = f1
        tdict["test_mean_dist"] = mean_dist

        return tdict
