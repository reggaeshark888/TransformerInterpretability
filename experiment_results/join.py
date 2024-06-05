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
        "output/join/join_weights.csv", index_col=[0, 1], dtype={"feature": str}
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(position, token):
        if position in {0, 2, 11, 12}:
            return token == "1"
        elif position in {1}:
            return token == "0"
        elif position in {3}:
            return token == "2"
        elif position in {4}:
            return token == "3"
        elif position in {5}:
            return token == "4"
        elif position in {6, 7, 8, 9, 10}:
            return token == "</s>"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {0}:
            return token == "0"
        elif position in {1, 2, 5}:
            return token == "3"
        elif position in {3, 4}:
            return token == "1"
        elif position in {6, 7, 8, 9, 10}:
            return token == "2"
        elif position in {11}:
            return token == "4"
        elif position in {12}:
            return token == "<pad>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(token):
        key = token
        if key in {"1", "2", "60000", "</s>"}:
            return 9
        elif key in {"0", "4", "69000"}:
            return 3
        elif key in {"62000"}:
            return 4
        return 6

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in tokens]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        if key in {8, 9, 10, 11}:
            return 10
        elif key in {0}:
            return 7
        return 11

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 11, 12}:
            return token == "<pad>"
        elif position in {1}:
            return token == "<s>"
        elif position in {2}:
            return token == "62000"
        elif position in {3}:
            return token == "3"
        elif position in {4}:
            return token == "80000"
        elif position in {5}:
            return token == "</s>"
        elif position in {9, 10, 6, 7}:
            return token == "2"
        elif position in {8}:
            return token == "4"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, num_mlp_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(position, token):
        if position in {0, 6, 7, 8, 9, 10}:
            return token == "<pad>"
        elif position in {1}:
            return token == "1"
        elif position in {2}:
            return token == "2"
        elif position in {3, 12}:
            return token == "4"
        elif position in {4, 5}:
            return token == "0"
        elif position in {11}:
            return token == "75000"

    num_attn_1_0_pattern = select(tokens, positions, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_0_output, token):
        key = (attn_0_0_output, token)
        if key in {
            (0, "3"),
            (0, "62000"),
            (0, "75000"),
            (0, "80000"),
            (0, "<s>"),
            (1, "<s>"),
            (2, "0"),
            (2, "2"),
            (2, "3"),
            (2, "<s>"),
            (3, "<s>"),
            (4, "1"),
            (4, "2"),
            (4, "<s>"),
            (5, "<s>"),
            (6, "<s>"),
            (7, "<s>"),
            (8, "<s>"),
            (9, "<s>"),
            (10, "<s>"),
            (11, "<s>"),
            (12, "0"),
            (12, "1"),
            (12, "2"),
            (12, "3"),
            (12, "62000"),
            (12, "75000"),
            (12, "80000"),
            (12, "</s>"),
            (12, "<s>"),
        }:
            return 7
        elif key in {
            (1, "3"),
            (3, "3"),
            (4, "3"),
            (4, "62000"),
            (4, "69000"),
            (4, "75000"),
            (4, "80000"),
            (4, "</s>"),
            (5, "3"),
            (6, "3"),
            (7, "3"),
            (7, "62000"),
            (7, "75000"),
            (8, "3"),
            (8, "62000"),
            (9, "3"),
            (9, "62000"),
            (10, "3"),
            (11, "3"),
        }:
            return 8
        elif key in {
            (0, "0"),
            (1, "0"),
            (1, "62000"),
            (1, "69000"),
            (1, "75000"),
            (1, "80000"),
            (1, "</s>"),
            (3, "0"),
            (4, "0"),
            (5, "0"),
            (6, "0"),
            (7, "0"),
            (7, "80000"),
            (8, "0"),
            (9, "0"),
            (9, "75000"),
            (9, "80000"),
            (10, "0"),
            (11, "0"),
        }:
            return 4
        elif key in {
            (0, "4"),
            (0, "</s>"),
            (1, "4"),
            (2, "4"),
            (3, "4"),
            (4, "4"),
            (5, "4"),
            (5, "62000"),
            (5, "69000"),
            (5, "80000"),
            (5, "</s>"),
            (6, "4"),
            (7, "4"),
            (8, "4"),
            (9, "4"),
            (10, "4"),
            (11, "4"),
            (12, "4"),
            (12, "69000"),
        }:
            return 11
        elif key in {
            (0, "1"),
            (1, "1"),
            (2, "1"),
            (2, "62000"),
            (2, "69000"),
            (2, "75000"),
            (2, "80000"),
            (3, "1"),
            (5, "1"),
            (6, "1"),
            (6, "62000"),
            (6, "80000"),
            (7, "1"),
            (8, "1"),
            (9, "1"),
            (10, "1"),
            (11, "1"),
        }:
            return 10
        elif key in {
            (0, "60000"),
            (1, "60000"),
            (2, "60000"),
            (3, "60000"),
            (4, "60000"),
            (5, "60000"),
            (6, "60000"),
            (7, "60000"),
            (8, "60000"),
            (9, "60000"),
            (10, "60000"),
            (11, "60000"),
            (12, "60000"),
        }:
            return 6
        return 9

    mlp_1_0_outputs = [mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, tokens)]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output, num_attn_0_0_output):
        key = (num_attn_1_0_output, num_attn_0_0_output)
        if key in {
            (8, 0),
            (9, 0),
            (10, 0),
            (11, 0),
            (11, 1),
            (12, 0),
            (12, 1),
            (13, 0),
            (13, 1),
            (14, 0),
            (14, 1),
            (14, 2),
            (15, 0),
            (15, 1),
            (15, 2),
            (16, 0),
            (16, 1),
            (16, 2),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (23, 0),
            (23, 1),
            (23, 2),
            (23, 3),
            (23, 4),
            (23, 5),
        }:
            return 3
        return 1

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_0_0_outputs)
    ]
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
            "80000",
            "75000",
            "80000",
            "3",
            "1",
            "0",
            "2",
            "0",
            "</s>",
        ]
    )
)
