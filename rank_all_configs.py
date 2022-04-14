import os

from ranking_metric import get_ranking_args, get_ranking_score_from_args

if __name__ == "__main__":
    base_args = get_ranking_args()
    gt_file = "results_full100_tuneepochs100.csv"
    ranking_dir = "ranking"
    base_args.ground_truth_csv = os.path.join(ranking_dir, gt_file)
#     base_args.ranking_method = "l1"
    
    all_ranks_df = None
    for file in sorted(os.listdir("ranking")):
        if "csv" in file: # and file != "gt_file":
            base_args.predict_csv = os.path.join(ranking_dir, file)
            
            rank_score, pred_df = get_ranking_score_from_args(base_args)

            print(f"Ranking score for {file} is: {rank_score:.3f}")

            if all_ranks_df is None:
                all_ranks_df = pred_df
                all_ranks_df = all_ranks_df.drop(columns=["val_accuracy"])
            exp_name = file.split("results_")[-1].split(".csv")[0]
            all_ranks_df[exp_name] = pred_df["val_accuracy"]
                    
    print(all_ranks_df)
    all_ranks_df.to_csv("all_ranks.csv")