import pandas as pd
import argparse
import numpy as np
from scipy import stats

pd.set_option('display.max_columns', None)

def ranking_score_kendalltau(gt_scores, pred_scores):
    """Calculate Kendall’s tau, a correlation measure for ordinal data.
    measure of the correspondence between two rankings.
    Values close to 1 indicate strong agreement,
    and values close to -1 indicate strong disagreement
    reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
    """
    gt_rank = gt_scores.argsort()
    pred_rank = pred_scores.argsort()
    correlation, p_value = stats.kendalltau(gt_rank, pred_rank)
    return correlation


def ranking_score_l1(gt_scores, pred_scores):
    gt_rank = gt_scores.argsort()
    pred_rank = pred_scores.argsort()
    correlation = 1 - np.abs(gt_rank - pred_rank).mean() / len(gt_rank)
    return correlation
    


def get_ranking_args():
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-gt', '--ground_truth_csv',
                        default="ranking/results_random20_epochs100.csv",
                        type=str,
                        help='ground truth csv file path (default: None)')
    parser.add_argument('-pred', '--predict_csv',
                        default="ranking/results_gradientcover20_epochs100.csv",
                        type=str,
                        help='predict csv file path (default: None)')
    parser.add_argument('--exp_fields',
                        default=[
                            'config/optimizer;type',
                            'config/optimizer;args;lr',
                            'config/data_loader;args;batch_size'
                        ],
                        type=str,
                        nargs='+',
                        help='the field names used to identify each experiment in the csv file')
    parser.add_argument('--metric_name',
                        default='val_accuracy',
                        type=str,
                        help='the filed name used to measure score of each experiment in the csv file')
    parser.add_argument('--verbose', default=False, type=bool, help='print mean, sorted dfs')
    parser.add_argument('--ranking_method', default='kendalltau', type=str, help='ranking method name')
    args = parser.parse_args()
    return args

def get_ranking_score_from_args(args):
    gt_df = pd.read_csv(args.ground_truth_csv)
    pred_df = pd.read_csv(args.predict_csv)
    # the fields used to identify experiment
    exp_fields = args.exp_fields
    # the field used to measure experiment performance
    metric_name = args.metric_name
    # Average the metric values for the experiments using the exp_fields to group
    gt_df = gt_df.groupby(exp_fields)[metric_name].mean().reset_index(drop=False).sort_values(exp_fields).reset_index(drop=True)
    pred_df = pred_df.groupby(exp_fields)[metric_name].mean().reset_index(drop=False).sort_values(exp_fields).reset_index(drop=True)
    
    if args.verbose:
        print('Ground truth with file:', args.ground_truth_csv)
        print(gt_df.sort_values(metric_name), "\n")
        print('Predictions with file:', args.predict_csv)
        print(pred_df.sort_values(metric_name), "\n")
    
    # assert gt csv and pred csv have identical set of experiments
    pd.testing.assert_frame_equal(gt_df[exp_fields], pred_df[exp_fields])
    gt_scores = gt_df[metric_name].to_numpy()
    pred_scores = pred_df[metric_name].to_numpy()
    if args.ranking_method == "kendalltau":
        rank_score = ranking_score_kendalltau(gt_scores, pred_scores)
    elif args.ranking_method == "l1":
        rank_score = ranking_score_l1(gt_scores, pred_scores)
    else:
        raise NotImplementedError(f"Ranking method: {args.ranking_method} is not supported")
    return rank_score

if __name__ == '__main__':
    args = get_ranking_args()
    rank_score = get_ranking_score_from_args(args)
    print("Ranking score is {}".format(rank_score))


