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


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/join_smaller/join_smaller_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"<s>", "4"}:
            return k_token == "4"
        elif q_token in {"5", "</s>"}:
            return k_token == "5"
        elif q_token in {"80000", "75000", "60000"}:
            return k_token == ""

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0, 8}:
            return token == "<s>"
        elif position in {1}:
            return token == "1"
        elif position in {2}:
            return token == "2"
        elif position in {3, 4}:
            return token == "0"
        elif position in {5, 6}:
            return token == "3"
        elif position in {10, 7}:
            return token == "80000"
        elif position in {9, 11}:
            return token == "</s>"
        elif position in {12}:
            return token == "4"
        elif position in {13}:
            return token == ""

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(token):
        key = token
        if key in {"", "60000", "</s>"}:
            return 9
        elif key in {"0", "3"}:
            return 3
        elif key in {"2"}:
            return 2
        elif key in {"1"}:
            return 5
        elif key in {"5"}:
            return 7
        return 8

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in tokens]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 8

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 7, 8, 9, 13}:
            return token == ""
        elif position in {1}:
            return token == "0"
        elif position in {2}:
            return token == "1"
        elif position in {3}:
            return token == "2"
        elif position in {4}:
            return token == "3"
        elif position in {5}:
            return token == "4"
        elif position in {6}:
            return token == "5"
        elif position in {10}:
            return token == "<s>"
        elif position in {11, 12}:
            return token == "</s>"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, attn_0_0_output):
        if position in {0}:
            return attn_0_0_output == 13
        elif position in {1}:
            return attn_0_0_output == 1
        elif position in {2}:
            return attn_0_0_output == 2
        elif position in {3}:
            return attn_0_0_output == 3
        elif position in {4, 13}:
            return attn_0_0_output == 4
        elif position in {5}:
            return attn_0_0_output == 5
        elif position in {6}:
            return attn_0_0_output == 6
        elif position in {7, 9, 10, 11, 12}:
            return attn_0_0_output == 10
        elif position in {8}:
            return attn_0_0_output == 7

    num_attn_1_0_pattern = select(attn_0_0_outputs, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(token, attn_1_0_output):
        key = (token, attn_1_0_output)
        if key in {
            ("0", 0),
            ("0", 1),
            ("0", 2),
            ("0", 3),
            ("0", 4),
            ("0", 5),
            ("0", 6),
            ("0", 7),
            ("0", 8),
            ("0", 9),
            ("0", 10),
            ("0", 11),
            ("0", 12),
            ("0", 13),
            ("2", 1),
            ("3", 1),
            ("4", 1),
            ("5", 1),
            ("75000", 1),
            ("75000", 8),
            ("80000", 1),
            ("</s>", 1),
            ("<s>", 0),
            ("<s>", 1),
            ("<s>", 7),
            ("<s>", 8),
            ("<s>", 9),
            ("<s>", 10),
            ("<s>", 12),
            ("<s>", 13),
        }:
            return 10
        elif key in {
            ("5", 0),
            ("5", 3),
            ("5", 4),
            ("5", 6),
            ("5", 8),
            ("5", 9),
            ("5", 10),
            ("5", 12),
            ("5", 13),
            ("75000", 6),
            ("75000", 7),
            ("75000", 9),
            ("80000", 0),
            ("80000", 6),
            ("80000", 7),
            ("80000", 8),
            ("80000", 9),
            ("80000", 10),
            ("80000", 12),
            ("80000", 13),
            ("</s>", 6),
            ("</s>", 12),
            ("<s>", 6),
        }:
            return 12
        elif key in {
            ("2", 0),
            ("2", 2),
            ("2", 3),
            ("2", 4),
            ("2", 5),
            ("2", 6),
            ("2", 8),
            ("2", 9),
            ("2", 10),
            ("2", 11),
            ("2", 12),
            ("2", 13),
            ("3", 0),
            ("3", 2),
            ("3", 3),
            ("3", 5),
            ("3", 6),
            ("3", 10),
            ("3", 11),
            ("5", 11),
            ("75000", 11),
            ("</s>", 3),
            ("</s>", 11),
            ("<s>", 3),
            ("<s>", 4),
            ("<s>", 11),
        }:
            return 9
        elif key in {
            ("2", 7),
            ("3", 7),
            ("5", 7),
            ("60000", 0),
            ("60000", 1),
            ("60000", 2),
            ("60000", 3),
            ("60000", 4),
            ("60000", 5),
            ("60000", 6),
            ("60000", 7),
            ("60000", 8),
            ("60000", 9),
            ("60000", 10),
            ("60000", 11),
            ("60000", 12),
            ("60000", 13),
            ("80000", 11),
        }:
            return 13
        elif key in {
            ("3", 4),
            ("3", 8),
            ("3", 9),
            ("3", 12),
            ("3", 13),
            ("75000", 3),
            ("75000", 4),
            ("75000", 12),
            ("75000", 13),
            ("80000", 3),
            ("80000", 4),
            ("</s>", 4),
        }:
            return 1
        return 3

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(tokens, attn_1_0_outputs)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 1

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                mlp_0_0_output_scores,
                num_mlp_0_0_output_scores,
                attn_1_0_output_scores,
                mlp_1_0_output_scores,
                num_mlp_1_0_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_1_0_output_scores,
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
            "<s>",
            "80000",
            "80000",
            "60000",
            "75000",
            "75000",
            "80000",
            "5",
            "4",
            "1",
            "1",
            "3",
            "0",
            "</s>",
        ]
    )
)
