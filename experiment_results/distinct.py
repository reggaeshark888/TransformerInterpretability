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
        "output/distinct/distinct_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(position, token):
        if position in {0, 4, 5, 7}:
            return token == "5"
        elif position in {1, 2, 3}:
            return token == "3"
        elif position in {6}:
            return token == "4"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 1, 3, 4, 5, 6, 7}:
            return k_position == 6
        elif q_position in {2}:
            return k_position == 5

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"3", "2"}:
            return k_token == "4"
        elif q_token in {"5", "4", "</s>"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "3"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"1"}:
            return k_token == "3"
        elif q_token in {"<s>", "2", "4", "3", "</s>"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "</s>"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, attn_0_2_output):
        key = (attn_0_3_output, attn_0_2_output)
        if key in {
            ("1", "5"),
            ("2", "5"),
            ("3", "5"),
            ("4", "5"),
            ("5", "1"),
            ("5", "2"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("5", "</s>"),
            ("5", "<s>"),
            ("</s>", "5"),
            ("<s>", "5"),
        }:
            return 5
        elif key in {
            ("1", "4"),
            ("2", "4"),
            ("4", "1"),
            ("4", "2"),
            ("4", "4"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("</s>", "4"),
            ("<s>", "4"),
        }:
            return 0
        elif key in {
            ("1", "3"),
            ("2", "3"),
            ("3", "1"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("</s>", "3"),
            ("<s>", "3"),
        }:
            return 3
        elif key in {("1", "2"), ("2", "1"), ("2", "2"), ("2", "</s>"), ("2", "<s>")}:
            return 2
        elif key in {("3", "2"), ("3", "3"), ("3", "4"), ("4", "3")}:
            return 7
        return 1

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_2_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 5}:
            return token == "3"
        elif mlp_0_0_output in {1, 4, 7}:
            return token == "4"
        elif mlp_0_0_output in {2, 6}:
            return token == "5"
        elif mlp_0_0_output in {3}:
            return token == "</s>"

    attn_1_0_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_1_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"5", "2", "4", "1"}:
            return k_token == "</s>"
        elif q_token in {"3", "</s>", "<s>"}:
            return k_token == "5"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_0_output, position):
        if attn_0_0_output in {"<s>", "2", "4", "1", "3", "5", "</s>"}:
            return position == 5

    attn_1_2_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_3_output, attn_0_2_output):
        if attn_0_3_output in {"3", "5", "4", "1"}:
            return attn_0_2_output == "5"
        elif attn_0_3_output in {"2"}:
            return attn_0_2_output == "<s>"
        elif attn_0_3_output in {"</s>"}:
            return attn_0_2_output == "2"
        elif attn_0_3_output in {"<s>"}:
            return attn_0_2_output == "4"

    attn_1_3_pattern = select_closest(attn_0_2_outputs, attn_0_3_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, mlp_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_2_output, attn_1_0_output):
        key = (attn_1_2_output, attn_1_0_output)
        if key in {
            ("1", "4"),
            ("1", "5"),
            ("2", "4"),
            ("3", "4"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("5", "1"),
            ("5", "4"),
            ("5", "5"),
            ("</s>", "4"),
            ("<s>", "4"),
            ("<s>", "5"),
        }:
            return 1
        elif key in {
            ("1", "3"),
            ("2", "3"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "5"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("5", "3"),
            ("</s>", "3"),
            ("<s>", "3"),
        }:
            return 7
        return 6

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_0_outputs)
    ]
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
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
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


print(run(["<s>", "1", "1", "1", "2", "</s>"]))
