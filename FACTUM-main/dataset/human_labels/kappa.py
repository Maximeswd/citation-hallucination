import pandas as pd
from sklearn.metrics import cohen_kappa_score


def calculate_stats():
    df = pd.read_csv("human_annotations.csv")

    raters = df["rater"].unique()
    models = df["model"].unique()

    print(f"{'=' * 60}")
    print(f"SUMMARY STATISTICS")
    print(f"Found raters: {raters}")
    print(f"Found models: {models}")
    print(f"{'=' * 60}")

    # Human vs LLM Agreement (per rater & per model)
    print("\n[1] HUMAN VS LLM KAPPA (Accuracy of LLM labels)")
    for rater in raters:
        print(f"\n--- Rater: {rater} ---")
        # Global for this rater
        subset = df[df["rater"] == rater]
        if len(subset) > 0:
            kappa_total = cohen_kappa_score(subset["label"], subset["llm_label"])
            print(f"  TOTAL vs LLM: {kappa_total:.4f} ({len(subset)} samples)")

        # Per model for this rater
        for model in models:
            m_subset = subset[subset["model"] == model]
            if len(m_subset) > 0:
                kappa_model = cohen_kappa_score(
                    m_subset["label"], m_subset["llm_label"]
                )
                print(
                    f"  Model {model:15}: {kappa_model:.4f} ({len(m_subset)} samples)"
                )

    # Inter-Rater Reliability (Human vs. Human)
    if len(raters) >= 2:
        print(f"\n\n{'=' * 60}")
        print("[2] INTER-RATER KAPPA (Human vs Human Agreement)")

        pivot_df = df.pivot(index="uid", columns="rater", values="label")
        uid_to_model = df[["uid", "model"]].drop_duplicates().set_index("uid")["model"]

        for i in range(len(raters)):
            for j in range(i + 1, len(raters)):
                r1, r2 = raters[i], raters[j]

                # Global Inter-rater
                shared = pivot_df[[r1, r2]].dropna()

                if not shared.empty:
                    kappa_inter = cohen_kappa_score(shared[r1], shared[r2])
                    print(f"\n--- {r1} vs {r2} ---")
                    print(
                        f"  GLOBAL Inter-Rater: {kappa_inter:.4f} ({len(shared)} shared samples)"
                    )

                    # Per Model Inter-rater
                    for model in models:
                        # Filter shared samples by those belonging to the current model
                        model_uids = uid_to_model[uid_to_model == model].index
                        model_shared = shared[shared.index.isin(model_uids)]

                        if len(model_shared) > 1:
                            # Cohen's kappa can't be calculated if all labels are the same
                            try:
                                k_model = cohen_kappa_score(
                                    model_shared[r1], model_shared[r2]
                                )
                                print(
                                    f"  Model {model:15}: {k_model:.4f} ({len(model_shared)} samples)"
                                )
                            except Exception:
                                print(
                                    f"  Model {model:15}: Error (likely zero variance in labels)"
                                )
                        elif len(model_shared) == 1:
                            print(f"  Model {model:15}: Too few shared samples (1)")
                else:
                    print(f"\nNo shared samples found between {r1} and {r2}.")


if __name__ == "__main__":
    try:
        calculate_stats()
    except FileNotFoundError:
        print("Error: 'human_annotations.csv' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
