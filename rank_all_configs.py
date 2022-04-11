import os

from ranking_metric import get_ranking_args, get_ranking_score_from_args

if __name__ == "__main__":
    base_args = get_ranking_args()
    gt_file = "results_full100_epochs100.csv"
    ranking_dir = "ranking"
    base_args.ground_truth_csv = os.path.join(ranking_dir, gt_file)
    
    for file in os.listdir("ranking"):
        if "csv" in file and file != "gt_file":
            base_args.predict_csv = os.path.join(ranking_dir, file)
            try:
                rank_score = get_ranking_score_from_args(base_args)

                print(f"Ranking score for {file} is {rank_score}")
            except Exception as e:
                print(e)