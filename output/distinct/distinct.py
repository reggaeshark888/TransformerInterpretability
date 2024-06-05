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
    def predicate_0_0(q_token, k_token):
        if q_token in {"1", "4", "3", "5", "2"}:
            return k_token == "6"
        elif q_token in {"9", "10"}:
            return k_token == "10"
        elif q_token in {"11"}:
            return k_token == "11"
        elif q_token in {"12"}:
            return k_token == "12"
        elif q_token in {"8", "7", "6"}:
            return k_token == "8"
        elif q_token in {"<BOS>"}:
            return k_token == "5"
        elif q_token in {"<EOS>"}:
            return k_token == "9"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(token, position):
        if token in {"1", "4", "12", "3", "7", "5", "6", "2"}:
            return position == 14
        elif token in {"8", "11", "9", "10"}:
            return position == 15
        elif token in {"<BOS>"}:
            return position == 12
        elif token in {"<EOS>"}:
            return position == 13

    attn_0_1_pattern = select_closest(positions, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, token):
        key = (attn_0_0_output, token)
        if key in {
            ("1", "1"),
            ("1", "12"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("1", "6"),
            ("12", "1"),
            ("12", "2"),
            ("12", "3"),
            ("12", "4"),
            ("12", "5"),
            ("12", "6"),
            ("2", "1"),
            ("2", "12"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "5"),
            ("2", "6"),
            ("2", "9"),
            ("3", "1"),
            ("3", "12"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "5"),
            ("3", "6"),
            ("3", "9"),
            ("4", "1"),
            ("4", "12"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "5"),
            ("4", "6"),
            ("4", "9"),
            ("5", "1"),
            ("5", "12"),
            ("5", "2"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("5", "6"),
            ("5", "9"),
            ("6", "1"),
            ("6", "12"),
            ("6", "2"),
            ("6", "3"),
            ("6", "4"),
            ("6", "5"),
            ("6", "6"),
            ("6", "9"),
            ("<BOS>", "1"),
            ("<BOS>", "12"),
            ("<BOS>", "2"),
            ("<BOS>", "3"),
            ("<BOS>", "4"),
            ("<BOS>", "5"),
            ("<BOS>", "6"),
            ("<BOS>", "9"),
            ("<EOS>", "1"),
            ("<EOS>", "12"),
            ("<EOS>", "2"),
            ("<EOS>", "3"),
            ("<EOS>", "4"),
            ("<EOS>", "5"),
            ("<EOS>", "6"),
            ("<EOS>", "9"),
        }:
            return 7
        elif key in {
            ("1", "10"),
            ("1", "11"),
            ("10", "1"),
            ("10", "10"),
            ("10", "11"),
            ("10", "12"),
            ("10", "2"),
            ("10", "3"),
            ("10", "4"),
            ("10", "5"),
            ("10", "6"),
            ("10", "7"),
            ("10", "8"),
            ("10", "9"),
            ("10", "<BOS>"),
            ("10", "<EOS>"),
            ("11", "1"),
            ("11", "10"),
            ("11", "11"),
            ("11", "12"),
            ("11", "2"),
            ("11", "3"),
            ("11", "4"),
            ("11", "5"),
            ("11", "6"),
            ("11", "7"),
            ("11", "8"),
            ("11", "9"),
            ("12", "10"),
            ("12", "11"),
            ("12", "12"),
            ("12", "7"),
            ("12", "8"),
            ("12", "9"),
            ("2", "10"),
            ("2", "11"),
            ("3", "10"),
            ("3", "11"),
            ("4", "10"),
            ("4", "11"),
            ("5", "10"),
            ("5", "11"),
            ("6", "10"),
            ("6", "11"),
            ("7", "10"),
            ("7", "11"),
            ("8", "10"),
            ("9", "10"),
            ("<BOS>", "10"),
            ("<BOS>", "11"),
            ("<EOS>", "10"),
            ("<EOS>", "11"),
        }:
            return 6
        elif key in {
            ("1", "7"),
            ("2", "7"),
            ("3", "7"),
            ("4", "7"),
            ("5", "7"),
            ("6", "7"),
            ("7", "1"),
            ("7", "12"),
            ("7", "2"),
            ("7", "3"),
            ("7", "4"),
            ("7", "5"),
            ("7", "6"),
            ("7", "7"),
            ("7", "8"),
            ("7", "9"),
            ("<BOS>", "7"),
        }:
            return 9
        return 16

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, tokens)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(mlp_0_0_output, token):
        if mlp_0_0_output in {0, 3, 4, 5, 8, 10, 13, 14, 15}:
            return token == ""
        elif mlp_0_0_output in {1}:
            return token == "6"
        elif mlp_0_0_output in {2, 11}:
            return token == "10"
        elif mlp_0_0_output in {6}:
            return token == "11"
        elif mlp_0_0_output in {7}:
            return token == "7"
        elif mlp_0_0_output in {16, 9}:
            return token == "9"
        elif mlp_0_0_output in {12}:
            return token == "12"

    attn_1_0_pattern = select_closest(tokens, mlp_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(mlp_0_0_output, position):
        if mlp_0_0_output in {0}:
            return position == 6
        elif mlp_0_0_output in {1, 2, 3, 4, 6, 7, 8, 12, 13, 16}:
            return position == 14
        elif mlp_0_0_output in {5, 9, 10, 11, 14, 15}:
            return position == 15

    attn_1_1_pattern = select_closest(positions, mlp_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output, attn_1_1_output):
        key = (attn_1_0_output, attn_1_1_output)
        if key in {
            ("1", "10"),
            ("1", "11"),
            ("1", "9"),
            ("10", "1"),
            ("10", "10"),
            ("10", "11"),
            ("10", "12"),
            ("10", "2"),
            ("10", "3"),
            ("10", "4"),
            ("10", "5"),
            ("10", "6"),
            ("10", "7"),
            ("10", "8"),
            ("10", "9"),
            ("10", "<BOS>"),
            ("10", "<EOS>"),
            ("11", "1"),
            ("11", "10"),
            ("11", "11"),
            ("11", "2"),
            ("11", "3"),
            ("11", "4"),
            ("11", "5"),
            ("11", "7"),
            ("11", "9"),
            ("11", "<EOS>"),
            ("12", "10"),
            ("12", "9"),
            ("2", "10"),
            ("2", "11"),
            ("2", "9"),
            ("3", "10"),
            ("3", "11"),
            ("3", "9"),
            ("4", "10"),
            ("4", "11"),
            ("4", "9"),
            ("5", "10"),
            ("5", "11"),
            ("5", "9"),
            ("6", "10"),
            ("6", "11"),
            ("6", "9"),
            ("7", "10"),
            ("7", "11"),
            ("7", "9"),
            ("8", "10"),
            ("8", "9"),
            ("9", "1"),
            ("9", "10"),
            ("9", "11"),
            ("9", "12"),
            ("9", "2"),
            ("9", "3"),
            ("9", "4"),
            ("9", "5"),
            ("9", "6"),
            ("9", "7"),
            ("9", "8"),
            ("9", "9"),
            ("9", "<BOS>"),
            ("9", "<EOS>"),
            ("<BOS>", "10"),
            ("<BOS>", "11"),
            ("<BOS>", "9"),
            ("<EOS>", "9"),
        }:
            return 16
        elif key in {
            ("1", "6"),
            ("1", "7"),
            ("11", "6"),
            ("12", "6"),
            ("12", "7"),
            ("2", "6"),
            ("2", "7"),
            ("3", "6"),
            ("3", "7"),
            ("4", "6"),
            ("4", "7"),
            ("5", "6"),
            ("5", "7"),
            ("6", "1"),
            ("6", "12"),
            ("6", "2"),
            ("6", "3"),
            ("6", "4"),
            ("6", "5"),
            ("6", "6"),
            ("6", "7"),
            ("6", "<BOS>"),
            ("6", "<EOS>"),
            ("7", "1"),
            ("7", "12"),
            ("7", "2"),
            ("7", "3"),
            ("7", "4"),
            ("7", "5"),
            ("7", "6"),
            ("7", "7"),
            ("7", "<EOS>"),
            ("<BOS>", "6"),
            ("<BOS>", "7"),
            ("<EOS>", "6"),
            ("<EOS>", "7"),
        }:
            return 0
        elif key in {
            ("1", "12"),
            ("1", "8"),
            ("11", "12"),
            ("11", "8"),
            ("12", "11"),
            ("12", "12"),
            ("12", "2"),
            ("12", "3"),
            ("12", "8"),
            ("12", "<EOS>"),
            ("2", "12"),
            ("2", "8"),
            ("3", "12"),
            ("3", "8"),
            ("4", "12"),
            ("4", "8"),
            ("5", "12"),
            ("5", "8"),
            ("6", "8"),
            ("7", "8"),
            ("8", "1"),
            ("8", "11"),
            ("8", "12"),
            ("8", "2"),
            ("8", "3"),
            ("8", "4"),
            ("8", "5"),
            ("8", "6"),
            ("8", "7"),
            ("8", "8"),
            ("8", "<BOS>"),
            ("8", "<EOS>"),
            ("<BOS>", "8"),
            ("<EOS>", "8"),
        }:
            return 5
        elif key in {
            ("1", "5"),
            ("12", "5"),
            ("2", "5"),
            ("3", "5"),
            ("4", "5"),
            ("5", "1"),
            ("5", "2"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("5", "<BOS>"),
            ("5", "<EOS>"),
            ("<BOS>", "5"),
            ("<EOS>", "5"),
        }:
            return 11
        return 1

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_1_outputs)
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


print(
    run(
        [
            "<BOS>",
            "1",
            "1",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "3",
            "3",
            "3",
            "4",
            "<EOS>",
        ]
    )
)
