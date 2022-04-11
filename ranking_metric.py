import pandas as pd
import argparse
from scipy import stats


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
    import pdb
    pdb.set_trace()
    
    gt_df.groupby(exp_fields)[metric_name].mean()
    
    
    gt_df = gt_df.sort_values(exp_fields).reset_index(drop=True)
    pred_df = pred_df.sort_values(exp_fields).reset_index(drop=True)
    # assert gt csv and pred csv have identical set of experiments
    pd.testing.assert_frame_equal(gt_df[exp_fields], pred_df[exp_fields])
    gt_scores = gt_df[metric_name].to_numpy()
    pred_scores = pred_df[metric_name].to_numpy()
    rank_score = ranking_score_kendalltau(gt_scores, pred_scores)
    return rank_score

if __name__ == '__main__':
    args = get_ranking_args()
    rank_score = get_ranking_score_from_args(args)
    print("Ranking score is {}".format(rank_score))


