import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/where/where_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(position, token):
        if position in {0, 1, 4, 5, 7, 8}:
            return token == "5"
        elif position in {2}:
            return token == "4"
        elif position in {3}:
            return token == "3"
        elif position in {10, 6}:
            return token == "2"
        elif position in {9}:
            return token == "1"
        elif position in {11}:
            return token == "8"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 1, 2, 3, 4, 5}:
            return k_position == 1
        elif q_position in {6}:
            return k_position == 10
        elif q_position in {7}:
            return k_position == 5
        elif q_position in {8, 9, 10}:
            return k_position == 2
        elif q_position in {11}:
            return k_position == 11

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(token):
        key = token
        if key in {"6", "</s>", "<s>"}:
            return 10
        return 5

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in tokens]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_1_output, token):
        if attn_0_1_output in {"4", "1"}:
            return token == "7"
        elif attn_0_1_output in {"2"}:
            return token == "8"
        elif attn_0_1_output in {"3"}:
            return token == "9"
        elif attn_0_1_output in {"5"}:
            return token == "</s>"
        elif attn_0_1_output in {"6"}:
            return token == "3"
        elif attn_0_1_output in {"7"}:
            return token == "<s>"
        elif attn_0_1_output in {"8"}:
            return token == "5"
        elif attn_0_1_output in {"9"}:
            return token == "1"
        elif attn_0_1_output in {"<s>", "</s>"}:
            return token == "<pad>"

    attn_1_0_pattern = select_closest(tokens, attn_0_1_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, token):
        if position in {0, 11}:
            return token == "<s>"
        elif position in {1, 2, 4, 5, 9}:
            return token == "5"
        elif position in {8, 3}:
            return token == "3"
        elif position in {6}:
            return token == "2"
        elif position in {10, 7}:
            return token == "1"

    attn_1_1_pattern = select_closest(tokens, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, positions)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(token):
        key = token
        if key in {"6"}:
            return 7
        return 11

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in tokens]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                mlp_0_0_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                mlp_1_0_output_scores,
                one_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "2", "3", "5", "1", "4", "9", "7", "6", "6", "8", "</s>"]))
