from typing import Dict
import os
import sys

import torch
import ipdb
from tqdm import tqdm

# Add the parent directory of the current script to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(parent_dir)

from geotransformer.engine.base_tester import BaseTester
from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.timer import Timer
from geotransformer.utils.common import get_log_string
from geotransformer.utils.torch import release_cuda, to_cuda

from registration.geotransformer_handling import fact_prediction, setup_params_for_fact


class SingleTester(BaseTester):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        super().__init__(cfg, parser=parser, cudnn_deterministic=cudnn_deterministic)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    def test_step(self, iteration, data_dict) -> Dict:
        pass

    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self):
        pass

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        return get_log_string(result_dict)

    def run(self):
        assert self.test_loader is not None
        self.load_snapshot(self.args.snapshot)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.before_test_epoch()
        summary_board = SummaryBoard(adaptive=True)
        timer = Timer()
        total_iterations = len(self.test_loader)
        pbar = tqdm(enumerate(self.test_loader), total=total_iterations)
        params_fact, self.args.fact_args, model_fact = setup_params_for_fact(self.args.fact_args)
        for iteration, data_dict in pbar:
            # NOTE (LUDDE): In data_dict['points'] we have the point clouds. I think perhaps the
            # first point cloud is not downsampled, and then they become sparser and sparser.
            # Perhaps they use superpoints or something similar to downsample the point clouds. Cause,
            # the batch_size is only 1. Note that data_dict['points'][0] contains the point cloud of
            # both the ref and the src point clouds. The ref point cloud is the first part (does not
            # need to be equally many in the two sets). See the forward method in the model.py file.
            # on start
            self.iteration = iteration + 1
            data_dict = to_cuda(data_dict)
            self.before_test_step(self.iteration, data_dict)
            # test step
            torch.cuda.synchronize()
            timer.add_prepare_time()
            output_dict = self.test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.add_process_time()
            # eval step
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            # logging
            result_dict = release_cuda(result_dict)
            summary_board.update_from_result_dict(result_dict)
            message = self.summary_string(self.iteration, data_dict, output_dict, result_dict)
            message += f', {timer.tostring()}'
            pbar.set_description(message)
            # NOTE: GT transform is in data_dict['transform'] and estimated transform is
            #       in output_dict['estimated_transform']
            # NOTE: In self.args.fact_args I can find the arguments for the FACT model

            # after step
            src = output_dict['src_points']
            ref = output_dict['ref_points']
            gt_trans = data_dict['transform']
            est_trans = output_dict['estimated_transform']
            del result_dict, output_dict, data_dict
            torch.cuda.empty_cache()

            fact_error_class = fact_prediction(self.args.fact_args, params_fact, model_fact,
                                               src, ref, est_trans, gt_trans)
            print(f"fact_error_class: {fact_error_class.item()} (0 means no error, 4 means high error)\n")
        self.after_test_epoch()
        summary_dict = summary_board.summary()
        message = get_log_string(result_dict=summary_dict, timer=timer)
        self.logger.critical(message)
