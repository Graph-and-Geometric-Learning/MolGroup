import os
import torch
import numpy as np
import pandas as pd
from ogb.graphproppred import Evaluator


# add the evaluation for qm8 and qm9
class OGBEvaluator(Evaluator):
    def __init__(self, name, root):
        
        self.name = name

        if name == "qm8":
            self.num_tasks = 16
            self.eval_metric = "mae"
        elif name == "qm9":
            self.num_tasks = 19
            self.eval_metric = "mae"
        elif name == "pcqm4mv2":
            self.num_tasks = 1
            self.eval_metric = "mae"
        else:
            meta_info = pd.read_csv(os.path.join(root, 'master.csv'), index_col=0, keep_default_na=False)
            if not self.name in meta_info:
                print(self.name)
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(meta_info.keys())
                raise ValueError(error_mssg)

            self.num_tasks = int(meta_info[self.name]['num tasks'])
            self.eval_metric = meta_info[self.name]['eval metric']
        
        # super().__init__(name)
    
    def eval(self, input_dict):

        if self.eval_metric == 'rocauc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rocauc(y_true, y_pred)
        if self.eval_metric == 'ap':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_ap(y_true, y_pred)
        elif self.eval_metric == 'rmse':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_rmse(y_true, y_pred)
        elif self.eval_metric == 'mae':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_mae(y_true, y_pred)
        elif self.eval_metric == 'acc':
            y_true, y_pred = self._parse_and_check_input(input_dict)
            return self._eval_acc(y_true, y_pred)
        elif self.eval_metric == 'F1':
            seq_ref, seq_pred = self._parse_and_check_input(input_dict)
            return self._eval_F1(seq_ref, seq_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == 'rocauc' or self.eval_metric == 'ap' or self.eval_metric == 'rmse' or self.eval_metric == 'acc' or self.eval_metric == 'mae':
            if not 'y_true' in input_dict:
                raise RuntimeError('Missing key of y_true')
            if not 'y_pred' in input_dict:
                raise RuntimeError('Missing key of y_pred')

            y_true, y_pred = input_dict['y_true'], input_dict['y_pred']

            '''
                y_true: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
                y_pred: numpy ndarray or torch tensor of shape (num_graphs, num_tasks)
            '''

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(y_true, torch.Tensor):
                y_true = y_true.detach().cpu().numpy()

            if torch is not None and isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.detach().cpu().numpy()


            ## check type
            if not isinstance(y_true, np.ndarray):
                raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

            if not y_true.shape == y_pred.shape:
                raise RuntimeError('Shape of y_true and y_pred must be the same')

            if not y_true.ndim == 2:
                raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

            if not y_true.shape[1] == self.num_tasks:
                raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.num_tasks, y_true.shape[1]))

            return y_true, y_pred

        elif self.eval_metric == 'F1':
            if not 'seq_ref' in input_dict:
                raise RuntimeError('Missing key of seq_ref')
            if not 'seq_pred' in input_dict:
                raise RuntimeError('Missing key of seq_pred')

            seq_ref, seq_pred = input_dict['seq_ref'], input_dict['seq_pred']

            if not isinstance(seq_ref, list):
                raise RuntimeError('seq_ref must be of type list')

            if not isinstance(seq_pred, list):
                raise RuntimeError('seq_pred must be of type list')

            if len(seq_ref) != len(seq_pred):
                raise RuntimeError('Length of seq_true and seq_pred should be the same')

            return seq_ref, seq_pred

        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    def _eval_mae(self, y_true, y_pred):
        '''
            compute RMSE score averaged across tasks
        '''
        mae_list = []

        for i in range(y_true.shape[1]):
            # ignore nan values
            is_labeled = y_true[:,i] == y_true[:,i]
            mae_list.append(np.mean(np.abs(y_true[is_labeled,i] - y_pred[is_labeled,i])))

        return {'mae': sum(mae_list)/len(mae_list)}