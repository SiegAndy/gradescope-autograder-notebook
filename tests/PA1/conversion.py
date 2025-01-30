import os
import pandas as pd
import numpy as np


def reorder_ranklsit(
    input_file,
    output_file,
):
    df = pd.read_csv(
        input_file,
        sep=" ",
        header=None,
        names=["query_id", "Q0", "doc_id", "rank", "score", "method"],
    )

    # Function to rescore within rank-based groups
    def rescore_group(group):
        # Define rank categories
        top_5 = group[group["rank"] <= 5]
        top_10 = group[(group["rank"] > 5) & (group["rank"] <= 10)]
        top_20 = group[(group["rank"] > 10) & (group["rank"] <= 20)]
        top_50 = group[(group["rank"] > 20) & (group["rank"] <= 50)]
        top_200 = group[(group["rank"] > 50) & (group["rank"] <= 200)]
        rest = group[group["rank"] > 200]

        def rescore_within(subgroup, start_rank):
            if subgroup.empty:
                return subgroup
            min_score, max_score = subgroup["score"].min(), subgroup["score"].max()
            subgroup["new_score"] = np.random.uniform(
                min_score, max_score, len(subgroup)
            )
            subgroup = subgroup.sort_values(
                by="new_score", ascending=False
            ).reset_index(drop=True)
            subgroup["rank"] = range(start_rank, start_rank + len(subgroup))
            subgroup["score"] = subgroup["new_score"]  # Assign new scores
            subgroup.drop(columns=["new_score"], inplace=True)
            return subgroup

        # Rescore within groups
        # top_5 = rescore_within(top_50, 1)
        # top_10 = rescore_within(top_50, 6)
        top_20 = rescore_within(top_50, 11)
        top_50 = rescore_within(top_200, 21)
        top_200 = rescore_within(top_200, 51)
        rest = rescore_within(rest, 201)

        return pd.concat([top_5, top_10, top_20, top_50, top_200, rest]).reset_index(
            drop=True
        )

    # Apply the function to each query group
    df = df.groupby("query_id", group_keys=False).apply(rescore_group)

    # Save the output in the same format
    df.to_csv(output_file, sep=" ", index=False, header=False)

    print(f"Updated ranklist saved to {output_file}")


output_dir = "new"
input_dir = "old"
file_list = [
    "msmarcosmall-bm25.trecrun",
    "msmarcosmall-dpr.trecrun",
    "msmarcosmall-ql.trecrun",
]
for file in file_list:
    reorder_ranklsit(os.path.join(input_dir, file), os.path.join(output_dir, file))
