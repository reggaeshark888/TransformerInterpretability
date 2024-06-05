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
    def predicate_0_0(token, position):
        if token in {"1", "11", "10", "12", "9"}:
            return position == 12
        elif token in {"2", "5", "8"}:
            return position == 10
        elif token in {"3"}:
            return position == 9
        elif token in {"4", "6", "7"}:
            return position == 11
        elif token in {"<BOS>"}:
            return position == 4
        elif token in {"<EOS>"}:
            return position == 0

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(token, position):
        if token in {"1"}:
            return position == 5
        elif token in {"9", "11", "10", "12"}:
            return position == 12
        elif token in {"2", "6", "<BOS>", "4"}:
            return position == 9
        elif token in {"3", "5", "7"}:
            return position == 10
        elif token in {"8", "<EOS>"}:
            return position == 11

    attn_0_1_pattern = select_closest(positions, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 6
        elif q_position in {1, 2, 3, 4, 5, 6, 7}:
            return k_position == 8
        elif q_position in {8, 9, 11}:
            return k_position == 10
        elif q_position in {10}:
            return k_position == 9
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 12

    attn_0_2_pattern = select_closest(positions, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(token, position):
        if token in {"1"}:
            return position == 10
        elif token in {"6", "3", "11", "10", "4", "7", "8", "12", "9", "2", "5"}:
            return position == 12
        elif token in {"<BOS>"}:
            return position == 7
        elif token in {"<EOS>"}:
            return position == 11

    attn_0_3_pattern = select_closest(positions, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(token):
        key = token
        if key in {"7", "8", "<BOS>", "<EOS>", "<pad>"}:
            return 6
        elif key in {"5", "6"}:
            return 3
        elif key in {"11"}:
            return 10
        return 14

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in tokens]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_0_output, token):
        if mlp_0_0_output in {0}:
            return token == "<EOS>"
        elif mlp_0_0_output in {1, 2, 3, 5, 6, 8}:
            return token == "7"
        elif mlp_0_0_output in {10, 11, 4, 12}:
            return token == "9"
        elif mlp_0_0_output in {9, 13, 7}:
            return token == "8"
        elif mlp_0_0_output in {14}:
            return token == "5"

    attn_1_0_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"1", "3", "11", "10", "2"}:
            return position == 12
        elif token in {"9", "8", "12", "7"}:
            return position == 11
        elif token in {"6", "4", "5", "<EOS>"}:
            return position == 10
        elif token in {"<BOS>"}:
            return position == 9

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(mlp_0_0_output, position):
        if mlp_0_0_output in {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}:
            return position == 12
        elif mlp_0_0_output in {1}:
            return position == 11

    attn_1_2_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_3_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"8", "5", "1", "7"}:
            return position == 10
        elif token in {"11", "10"}:
            return position == 12
        elif token in {"4", "12", "9", "2", "6"}:
            return position == 11
        elif token in {"3"}:
            return position == 9
        elif token in {"<BOS>"}:
            return position == 13
        elif token in {"<EOS>"}:
            return position == 0

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_2_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, attn_1_0_output):
        key = (attn_1_3_output, attn_1_0_output)
        if key in {
            ("1", "6"),
            ("1", "7"),
            ("10", "6"),
            ("10", "7"),
            ("10", "9"),
            ("11", "6"),
            ("11", "7"),
            ("11", "8"),
            ("11", "9"),
            ("12", "6"),
            ("12", "7"),
            ("12", "8"),
            ("12", "9"),
            ("2", "6"),
            ("2", "7"),
            ("3", "6"),
            ("3", "7"),
            ("4", "6"),
            ("4", "7"),
            ("5", "6"),
            ("5", "7"),
            ("5", "8"),
            ("6", "1"),
            ("6", "10"),
            ("6", "11"),
            ("6", "12"),
            ("6", "2"),
            ("6", "3"),
            ("6", "4"),
            ("6", "5"),
            ("6", "6"),
            ("6", "7"),
            ("6", "8"),
            ("6", "9"),
            ("6", "<BOS>"),
            ("6", "<EOS>"),
            ("7", "1"),
            ("7", "10"),
            ("7", "11"),
            ("7", "12"),
            ("7", "2"),
            ("7", "3"),
            ("7", "4"),
            ("7", "5"),
            ("7", "6"),
            ("7", "7"),
            ("7", "8"),
            ("7", "9"),
            ("7", "<BOS>"),
            ("7", "<EOS>"),
            ("8", "10"),
            ("8", "11"),
            ("8", "12"),
            ("8", "6"),
            ("8", "7"),
            ("8", "8"),
            ("8", "9"),
            ("9", "1"),
            ("9", "10"),
            ("9", "11"),
            ("9", "12"),
            ("9", "2"),
            ("9", "5"),
            ("9", "6"),
            ("9", "7"),
            ("9", "8"),
            ("9", "9"),
            ("<BOS>", "6"),
            ("<BOS>", "7"),
            ("<EOS>", "6"),
            ("<EOS>", "7"),
        }:
            return 9
        elif key in {
            ("1", "4"),
            ("1", "8"),
            ("10", "4"),
            ("10", "8"),
            ("11", "4"),
            ("12", "4"),
            ("2", "4"),
            ("2", "8"),
            ("3", "4"),
            ("3", "8"),
            ("4", "1"),
            ("4", "10"),
            ("4", "11"),
            ("4", "12"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("4", "8"),
            ("4", "9"),
            ("4", "<BOS>"),
            ("4", "<EOS>"),
            ("5", "4"),
            ("8", "1"),
            ("8", "2"),
            ("8", "3"),
            ("8", "4"),
            ("8", "5"),
            ("8", "<BOS>"),
            ("8", "<EOS>"),
            ("9", "3"),
            ("9", "4"),
            ("<BOS>", "4"),
            ("<BOS>", "8"),
            ("<EOS>", "4"),
            ("<EOS>", "8"),
        }:
            return 2
        return 7

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_0_outputs)
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


print(run(["<BOS>", "1", "2", "2", "3", "4", "5", "6", "6", "7", "8", "9", "<EOS>"]))
