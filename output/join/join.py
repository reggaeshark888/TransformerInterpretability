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
        if position in {0, 1, 11, 12}:
            return token == "0"
        elif position in {2}:
            return token == "1"
        elif position in {3}:
            return token == "2"
        elif position in {4}:
            return token == "3"
        elif position in {5}:
            return token == "4"
        elif position in {8, 6}:
            return token == "<s>"
        elif position in {10, 7}:
            return token == "</s>"
        elif position in {9}:
            return token == "<pad>"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 1, 11}:
            return token == "0"
        elif position in {2}:
            return token == "1"
        elif position in {3}:
            return token == "2"
        elif position in {4}:
            return token == "3"
        elif position in {5}:
            return token == "4"
        elif position in {6, 7}:
            return token == "</s>"
        elif position in {8, 9, 10}:
            return token == "<s>"
        elif position in {12}:
            return token == "69000"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(position, token):
        if position in {0, 1, 11, 12}:
            return token == "0"
        elif position in {2}:
            return token == "1"
        elif position in {3}:
            return token == "2"
        elif position in {4}:
            return token == "3"
        elif position in {5}:
            return token == "4"
        elif position in {10, 6, 7}:
            return token == "<s>"
        elif position in {8, 9}:
            return token == "</s>"

    attn_0_2_pattern = select_closest(tokens, positions, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 1, 11, 12}:
            return token == "0"
        elif position in {2}:
            return token == "1"
        elif position in {3}:
            return token == "2"
        elif position in {4, 7, 8, 9, 10}:
            return token == "</s>"
        elif position in {5}:
            return token == "4"
        elif position in {6}:
            return token == "60000"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(token, attn_0_1_output):
        key = (token, attn_0_1_output)
        if key in {
            ("60000", "75000"),
            ("69000", "75000"),
            ("75000", "2"),
            ("75000", "69000"),
            ("75000", "80000"),
            ("75000", "</s>"),
            ("75000", "<s>"),
            ("80000", "2"),
            ("80000", "3"),
            ("80000", "60000"),
            ("80000", "62000"),
            ("80000", "69000"),
            ("80000", "75000"),
            ("80000", "80000"),
            ("80000", "</s>"),
            ("80000", "<s>"),
            ("</s>", "75000"),
            ("</s>", "80000"),
            ("</s>", "</s>"),
            ("<s>", "0"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "60000"),
            ("<s>", "62000"),
            ("<s>", "69000"),
            ("<s>", "75000"),
            ("<s>", "80000"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 10
        elif key in {
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "60000"),
            ("1", "62000"),
            ("1", "69000"),
            ("1", "75000"),
            ("1", "80000"),
            ("1", "</s>"),
            ("1", "<s>"),
            ("60000", "1"),
            ("69000", "1"),
            ("75000", "1"),
            ("75000", "62000"),
            ("80000", "1"),
            ("</s>", "1"),
            ("<s>", "1"),
        }:
            return 9
        elif key in {
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "60000"),
            ("3", "62000"),
            ("3", "69000"),
            ("3", "75000"),
            ("3", "80000"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("60000", "3"),
            ("62000", "3"),
            ("69000", "3"),
            ("75000", "3"),
            ("</s>", "3"),
        }:
            return 8
        elif key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "3"),
            ("0", "4"),
            ("0", "60000"),
            ("0", "62000"),
            ("0", "69000"),
            ("0", "75000"),
            ("0", "80000"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("60000", "0"),
            ("69000", "0"),
            ("75000", "0"),
            ("80000", "0"),
        }:
            return 5
        elif key in {
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "60000"),
            ("4", "62000"),
            ("4", "69000"),
            ("4", "80000"),
            ("4", "</s>"),
            ("4", "<s>"),
        }:
            return 12
        elif key in {
            ("4", "4"),
            ("4", "75000"),
            ("60000", "4"),
            ("69000", "4"),
            ("75000", "4"),
            ("75000", "60000"),
            ("75000", "75000"),
            ("80000", "4"),
        }:
            return 6
        return 1

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(tokens, attn_0_1_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0}:
            return token == "69000"
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
        elif position in {12, 6}:
            return token == "<s>"
        elif position in {8, 9, 10, 7}:
            return token == "</s>"
        elif position in {11}:
            return token == "<pad>"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(position, token):
        if position in {0, 5}:
            return token == "4"
        elif position in {1}:
            return token == "0"
        elif position in {2}:
            return token == "1"
        elif position in {3}:
            return token == "2"
        elif position in {4}:
            return token == "</s>"
        elif position in {6}:
            return token == "60000"
        elif position in {7}:
            return token == "62000"
        elif position in {8, 11, 12}:
            return token == "<pad>"
        elif position in {9}:
            return token == "75000"
        elif position in {10}:
            return token == "80000"

    attn_1_1_pattern = select_closest(tokens, positions, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, token):
        if position in {0, 5}:
            return token == "4"
        elif position in {1}:
            return token == "0"
        elif position in {2}:
            return token == "1"
        elif position in {8, 3, 6, 7}:
            return token == "2"
        elif position in {9, 4}:
            return token == "3"
        elif position in {10}:
            return token == "</s>"
        elif position in {11, 12}:
            return token == "<pad>"

    attn_1_2_pattern = select_closest(tokens, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(position, token):
        if position in {0}:
            return token == "62000"
        elif position in {1}:
            return token == "0"
        elif position in {8, 2}:
            return token == "1"
        elif position in {9, 10, 3, 6}:
            return token == "2"
        elif position in {4}:
            return token == "3"
        elif position in {5}:
            return token == "4"
        elif position in {7}:
            return token == "80000"
        elif position in {11, 12}:
            return token == "<pad>"

    attn_1_3_pattern = select_closest(tokens, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_0_output, attn_0_2_output):
        key = (attn_0_0_output, attn_0_2_output)
        if key in {
            ("0", "4"),
            ("1", "4"),
            ("2", "4"),
            ("3", "4"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("4", "4"),
            ("4", "60000"),
            ("4", "62000"),
            ("4", "69000"),
            ("4", "75000"),
            ("4", "80000"),
            ("4", "</s>"),
            ("4", "<s>"),
            ("60000", "4"),
            ("60000", "62000"),
            ("60000", "75000"),
            ("60000", "</s>"),
            ("60000", "<s>"),
            ("62000", "4"),
            ("69000", "4"),
            ("75000", "4"),
            ("75000", "62000"),
            ("75000", "75000"),
            ("80000", "4"),
            ("</s>", "4"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "60000"),
            ("<s>", "62000"),
            ("<s>", "69000"),
            ("<s>", "80000"),
            ("<s>", "</s>"),
            ("<s>", "<s>"),
        }:
            return 11
        elif key in {
            ("0", "3"),
            ("1", "3"),
            ("2", "3"),
            ("3", "0"),
            ("3", "1"),
            ("3", "2"),
            ("3", "3"),
            ("3", "60000"),
            ("3", "62000"),
            ("3", "69000"),
            ("3", "75000"),
            ("3", "80000"),
            ("3", "</s>"),
            ("3", "<s>"),
            ("60000", "3"),
            ("62000", "3"),
            ("62000", "60000"),
            ("62000", "69000"),
            ("62000", "75000"),
            ("62000", "80000"),
            ("62000", "</s>"),
            ("69000", "3"),
            ("75000", "3"),
            ("80000", "3"),
            ("</s>", "3"),
        }:
            return 3
        elif key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("0", "60000"),
            ("0", "62000"),
            ("0", "69000"),
            ("0", "75000"),
            ("0", "80000"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("1", "0"),
            ("2", "0"),
            ("60000", "0"),
            ("62000", "0"),
            ("62000", "62000"),
            ("62000", "<s>"),
            ("69000", "0"),
            ("75000", "0"),
            ("80000", "0"),
            ("80000", "62000"),
            ("80000", "<s>"),
            ("</s>", "0"),
            ("<s>", "0"),
        }:
            return 4
        elif key in {
            ("1", "60000"),
            ("60000", "1"),
            ("60000", "2"),
            ("60000", "60000"),
            ("60000", "69000"),
            ("60000", "80000"),
            ("69000", "60000"),
            ("75000", "60000"),
            ("80000", "60000"),
            ("</s>", "60000"),
            ("<s>", "75000"),
        }:
            return 9
        elif key in {("<s>", "1"), ("<s>", "2")}:
            return 1
        return 2

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_2_outputs)
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
