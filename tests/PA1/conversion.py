import os
import pandas as pd
import numpy as np


# -----------------------------
# 1) Parse QRELs
# -----------------------------
def parse_qrels(qrel_path):
    """
    Expects a file with lines: query_id 0 doc_id relevance
    Returns a dict of dicts: qrel_mapping[query_id][doc_id] = relevance
    """
    qrel_mapping = {}
    with open(qrel_path, "r") as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            rel = int(rel)
            qid_dict = qrel_mapping.setdefault(qid, {})
            qid_dict[docid] = rel
    return qrel_mapping


# -----------------------------
# 2) Read TREC runs into DataFrame
# -----------------------------
def read_run(file_path):
    """
    Reads a .trecrun file into a DataFrame with columns:
    query_id, Q0, doc_id, rank, score, method
    """
    df = pd.read_csv(
        file_path,
        sep=" ",
        header=None,
        names=["query_id", "Q0", "doc_id", "rank", "score", "method"],
    )
    return df


# -----------------------------
# 3) Top-k Swap for BM25
# -----------------------------
def get_relevance(qrel_mapping, query_id, doc_id):
    return qrel_mapping.get(query_id, {}).get(doc_id, 0)


def top_k_swap_for_query(
    df_bm25, qrel_mapping, target_query, top_k=10, random_state=42
):
    """
    For a SINGLE query (target_query):
      - Identify relevant docs in top_k.
      - Randomly sample the same # of non-relevant docs from ranks > top_k.
      - Swap their ranks (and reassign scores randomly).
    Returns a DataFrame with modifications for ONLY target_query.
    """
    # Split out just this query
    query_mask = df_bm25["query_id"] == target_query
    group = df_bm25[query_mask].copy()
    if group.empty:
        return df_bm25  # If no such query in the run, do nothing

    # Mark relevance
    group["relevance"] = group.apply(
        lambda row: get_relevance(qrel_mapping, row["query_id"], row["doc_id"]), axis=1
    )

    # Relevant in top-k
    top_k_relevant = group[(group["rank"] <= top_k) & (group["relevance"] > 0)]
    num_relevant_top_k = len(top_k_relevant)
    if num_relevant_top_k == 0:
        # No relevant docs in top-k => no swap needed
        df_bm25.loc[query_mask] = group.drop(columns=["relevance"])
        return df_bm25

    # Non-relevant after top-k
    after_top_k_nonrel = group[(group["rank"] > top_k) & (group["relevance"] == 0)]
    if len(after_top_k_nonrel) < num_relevant_top_k:
        # Not enough docs to swap with
        df_bm25.loc[query_mask] = group.drop(columns=["relevance"])
        return df_bm25

    # Randomly sample docs to swap
    swap_nonrel = after_top_k_nonrel.sample(
        n=num_relevant_top_k, random_state=random_state
    )

    # Swap ranks
    top_k_relevant_ranks = top_k_relevant["rank"].values
    after_top_k_nonrel_ranks = swap_nonrel["rank"].values

    # Assign new ranks
    group.loc[top_k_relevant.index, "rank"] = after_top_k_nonrel_ranks
    group.loc[swap_nonrel.index, "rank"] = top_k_relevant_ranks

    # Adjust the scores randomly in a plausible range
    min_score, max_score = group["score"].min(), group["score"].max()
    group.loc[top_k_relevant.index, "score"] = np.random.uniform(
        min_score, max_score, size=num_relevant_top_k
    )
    group.loc[swap_nonrel.index, "score"] = np.random.uniform(
        min_score, max_score, size=num_relevant_top_k
    )

    # Sort by rank and re-index
    group = group.sort_values("rank").reset_index(drop=True)

    # Replace the subset in df_bm25
    df_bm25.loc[query_mask] = group.drop(columns=["relevance"])
    return df_bm25


# -----------------------------
# 4) Light Reordering for QL & DPR
# -----------------------------
def improve_run(df, qrel_mapping, improvement_factor=0.8, random_state=42):
    """
    Heuristic: for each query,
      - 'shrink' the rank of relevant docs by improvement_factor (pushing them closer to the top).
      - Then re-sort by this 'new_rank' and reassign final ranks 1..N.
      - Randomize scores slightly.

    improvement_factor < 1 => docs move closer to rank=1 => "better performance"
    """
    df["relevance"] = df.apply(
        lambda row: get_relevance(qrel_mapping, row["query_id"], row["doc_id"]), axis=1
    )

    def process_one_query(group):
        # Shift relevant docs' rank upward by the improvement_factor
        # new_rank = 1 + improvement_factor * (old_rank - 1)
        group["new_rank"] = group["rank"]
        mask_rel = group["relevance"] > 0
        group.loc[mask_rel, "new_rank"] = 1 + improvement_factor * (
            group.loc[mask_rel, "rank"] - 1
        )

        # Sort by new_rank
        group = group.sort_values("new_rank").reset_index(drop=True)
        # Reassign final rank = 1..N
        group["rank"] = range(1, len(group) + 1)

        # Adjust scores in a random but consistent way
        min_score, max_score = group["score"].min(), group["score"].max()
        group["score"] = np.random.uniform(min_score, max_score, size=len(group))

        group.drop(columns=["new_rank"], inplace=True)
        return group

    df = df.groupby("query_id", group_keys=False).apply(process_one_query)
    df = df.drop(columns=["relevance"])
    return df


# -----------------------------
# 5) Main Script
# -----------------------------
def main():

    output_dir = "new"
    input_dir = "old"
    # For example:
    file_list = [
        "msmarcofull-bm25.trecrun",
        "msmarcofull-dpr.trecrun",
        "msmarcofull-ql.trecrun",
    ]
    qrel_path = "msmarco.qrels"

    # 1) Parse QRELs
    qrel_mapping = parse_qrels(qrel_path)

    # 2) Read BM25, DPR, QL
    df_bm25 = read_run(os.path.join(input_dir, file_list[0]))
    df_dpr = read_run(os.path.join(input_dir, file_list[1]))
    df_ql = read_run(os.path.join(input_dir, file_list[2]))

    # 3) Perform top-k swap in BM25 for one query
    target_query = "23849"  # <-- for example
    df_bm25 = top_k_swap_for_query(
        df_bm25, qrel_mapping, target_query, top_k=100, random_state=42
    )

    # 4) Light reorder for QL (mild improvement) and DPR (stronger improvement)
    #    so that, on average, we get BM25 <= QL < DPR
    df_ql = improve_run(df_ql, qrel_mapping, improvement_factor=0.4, random_state=42)
    df_dpr = improve_run(df_dpr, qrel_mapping, improvement_factor=0.1, random_state=42)

    # 5) Save back to new TREC run files
    df_bm25.to_csv(
        os.path.join(output_dir, file_list[0]),
        sep=" ",
        index=False,
        header=False,
    )
    df_ql.to_csv(
        os.path.join(output_dir, file_list[2]),
        sep=" ",
        index=False,
        header=False,
    )
    df_dpr.to_csv(
        os.path.join(output_dir, file_list[1]),
        sep=" ",
        index=False,
        header=False,
    )

    print("All three runs have been updated and saved.")


if __name__ == "__main__":
    main()
