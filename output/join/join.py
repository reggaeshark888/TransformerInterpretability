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
        if position in {0, 11, 3, 12}:
            return token == "2"
        elif position in {1}:
            return token == "0"
        elif position in {9, 2, 6}:
            return token == "1"
        elif position in {10, 4}:
            return token == "3"
        elif position in {5, 7}:
            return token == "4"
        elif position in {8}:
            return token == "<pad>"

    attn_0_0_pattern = select_closest(tokens, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(position, token):
        if position in {0, 1, 8, 9, 10, 12}:
            return token == "0"
        elif position in {2, 6}:
            return token == "1"
        elif position in {11, 3, 7}:
            return token == "2"
        elif position in {4}:
            return token == "3"
        elif position in {5}:
            return token == "4"

    attn_0_1_pattern = select_closest(tokens, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"</s>", "0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4", "<s>"}:
            return k_token == "4"
        elif q_token in {"60000"}:
            return k_token == "</s>"
        elif q_token in {"69000", "62000", "80000"}:
            return k_token == "<pad>"
        elif q_token in {"75000"}:
            return k_token == "<s>"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(position, token):
        if position in {0, 4}:
            return token == "3"
        elif position in {1}:
            return token == "0"
        elif position in {2, 11}:
            return token == "1"
        elif position in {3, 12}:
            return token == "2"
        elif position in {5}:
            return token == "4"
        elif position in {6, 7}:
            return token == "</s>"
        elif position in {8, 10}:
            return token == "<pad>"
        elif position in {9}:
            return token == "<s>"

    attn_0_3_pattern = select_closest(tokens, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(q_position, k_position):
        if q_position in {0, 2}:
            return k_position == 2
        elif q_position in {1, 11}:
            return k_position == 1
        elif q_position in {3, 12}:
            return k_position == 3
        elif q_position in {4}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 0
        elif q_position in {9, 7}:
            return k_position == 12
        elif q_position in {8, 10}:
            return k_position == 10

    attn_0_4_pattern = select_closest(positions, positions, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(position, token):
        if position in {0, 4, 6}:
            return token == "3"
        elif position in {1}:
            return token == "0"
        elif position in {2, 7}:
            return token == "1"
        elif position in {3, 9, 10, 11, 12}:
            return token == "2"
        elif position in {8, 5}:
            return token == "4"

    attn_0_5_pattern = select_closest(tokens, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output, token):
        key = (attn_0_0_output, token)
        if key in {
            ("0", "3"),
            ("1", "3"),
            ("1", "69000"),
            ("2", "3"),
            ("2", "69000"),
            ("3", "3"),
            ("3", "69000"),
            ("4", "3"),
            ("60000", "3"),
            ("60000", "69000"),
            ("62000", "3"),
            ("62000", "69000"),
            ("62000", "</s>"),
            ("69000", "3"),
            ("69000", "69000"),
            ("69000", "</s>"),
            ("75000", "3"),
            ("75000", "69000"),
            ("80000", "3"),
            ("80000", "69000"),
            ("</s>", "3"),
            ("</s>", "69000"),
            ("<s>", "3"),
            ("<s>", "69000"),
        }:
            return 8
        elif key in {
            ("0", "0"),
            ("0", "62000"),
            ("0", "69000"),
            ("1", "0"),
            ("1", "62000"),
            ("2", "0"),
            ("2", "62000"),
            ("3", "0"),
            ("3", "62000"),
            ("4", "0"),
            ("4", "62000"),
            ("4", "69000"),
            ("60000", "0"),
            ("60000", "62000"),
            ("62000", "0"),
            ("62000", "62000"),
            ("69000", "0"),
            ("69000", "62000"),
            ("75000", "0"),
            ("75000", "62000"),
            ("80000", "0"),
            ("80000", "62000"),
            ("</s>", "0"),
            ("</s>", "62000"),
            ("<s>", "0"),
            ("<s>", "62000"),
        }:
            return 5
        elif key in {
            ("1", "4"),
            ("2", "4"),
            ("3", "4"),
            ("60000", "4"),
            ("69000", "4"),
            ("80000", "4"),
            ("</s>", "4"),
            ("<s>", "4"),
        }:
            return 12
        elif key in {("0", "4"), ("4", "4"), ("62000", "4"), ("75000", "4")}:
            return 9
        elif key in {
            ("0", "75000"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("4", "75000"),
            ("4", "</s>"),
            ("4", "<s>"),
        }:
            return 2
        return 10

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, tokens)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, token):
        key = (attn_0_2_output, token)
        if key in {
            ("0", "1"),
            ("1", "1"),
            ("2", "1"),
            ("3", "1"),
            ("60000", "1"),
            ("62000", "1"),
            ("69000", "1"),
            ("75000", "1"),
            ("80000", "1"),
            ("</s>", "1"),
            ("<s>", "1"),
        }:
            return 6
        elif key in {("1", "2"), ("2", "2")}:
            return 4
        elif key in {("80000", "2")}:
            return 12
        return 11

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(position, token):
        if position in {0, 2, 7}:
            return token == "1"
        elif position in {1}:
            return token == "0"
        elif position in {10, 3}:
            return token == "2"
        elif position in {4, 6, 8, 9, 11}:
            return token == "3"
        elif position in {5}:
            return token == "4"
        elif position in {12}:
            return token == "<pad>"

    attn_1_0_pattern = select_closest(tokens, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, tokens)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"62000", "0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2", "75000"}:
            return k_token == "2"
        elif q_token in {"</s>", "69000", "3"}:
            return k_token == "3"
        elif q_token in {"4", "80000"}:
            return k_token == "4"
        elif q_token in {"60000"}:
            return k_token == "<pad>"
        elif q_token in {"<s>"}:
            return k_token == "62000"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_2_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"69000", "62000", "0", "75000"}:
            return k_token == "0"
        elif q_token in {"1", "</s>"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"60000", "3"}:
            return k_token == "3"
        elif q_token in {"4", "80000"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == "<pad>"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, tokens)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 6}:
            return k_position == 1
        elif q_position in {1}:
            return k_position == 11
        elif q_position in {2}:
            return k_position == 2
        elif q_position in {3}:
            return k_position == 6
        elif q_position in {9, 4, 7}:
            return k_position == 4
        elif q_position in {5}:
            return k_position == 5
        elif q_position in {8, 10, 11}:
            return k_position == 3
        elif q_position in {12}:
            return k_position == 8

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, tokens)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(position, token):
        if position in {0, 8, 10, 5}:
            return token == "4"
        elif position in {1, 12}:
            return token == "<s>"
        elif position in {2, 6}:
            return token == "1"
        elif position in {3}:
            return token == "2"
        elif position in {9, 4, 7}:
            return token == "3"
        elif position in {11}:
            return token == "0"

    attn_1_4_pattern = select_closest(tokens, positions, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, tokens)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(position, token):
        if position in {0, 4}:
            return token == "3"
        elif position in {1}:
            return token == "</s>"
        elif position in {2}:
            return token == "<s>"
        elif position in {8, 11, 3, 6}:
            return token == "2"
        elif position in {5}:
            return token == "4"
        elif position in {9, 7}:
            return token == "0"
        elif position in {10}:
            return token == "1"
        elif position in {12}:
            return token == "<pad>"

    attn_1_5_pattern = select_closest(tokens, positions, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, tokens)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_2_output, attn_1_0_output):
        key = (attn_0_2_output, attn_1_0_output)
        if key in {
            ("0", "1"),
            ("0", "3"),
            ("0", "4"),
            ("1", "2"),
            ("1", "3"),
            ("1", "4"),
            ("2", "1"),
            ("2", "3"),
            ("2", "4"),
            ("3", "0"),
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
            ("60000", "4"),
            ("60000", "69000"),
            ("60000", "75000"),
            ("62000", "4"),
            ("62000", "62000"),
            ("62000", "69000"),
            ("62000", "75000"),
            ("62000", "80000"),
            ("69000", "4"),
            ("75000", "4"),
            ("75000", "69000"),
            ("75000", "75000"),
            ("80000", "4"),
            ("</s>", "4"),
            ("<s>", "4"),
        }:
            return 8
        elif key in {("1", "<s>")}:
            return 2
        elif key in {("2", "<s>")}:
            return 5
        return 1

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_1_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_2_output, attn_1_0_output):
        key = (attn_1_2_output, attn_1_0_output)
        if key in {
            ("0", "0"),
            ("0", "2"),
            ("0", "60000"),
            ("0", "62000"),
            ("0", "69000"),
            ("0", "75000"),
            ("0", "80000"),
            ("0", "</s>"),
            ("0", "<s>"),
            ("2", "2"),
            ("2", "62000"),
            ("2", "75000"),
            ("2", "</s>"),
            ("60000", "2"),
            ("60000", "</s>"),
            ("62000", "2"),
            ("62000", "</s>"),
            ("69000", "2"),
            ("69000", "</s>"),
            ("75000", "0"),
            ("75000", "2"),
            ("75000", "62000"),
            ("75000", "69000"),
            ("75000", "75000"),
            ("75000", "</s>"),
            ("80000", "2"),
            ("<s>", "2"),
        }:
            return 10
        elif key in {("2", "0"), ("4", "2"), ("</s>", "2")}:
            return 11
        return 7

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(position, token):
        if position in {0, 10, 12}:
            return token == "<s>"
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
        elif position in {8, 11, 6, 7}:
            return token == "<pad>"
        elif position in {9}:
            return token == "60000"

    attn_2_0_pattern = select_closest(tokens, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, mlp_1_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(position, token):
        if position in {0, 3, 7, 10, 11, 12}:
            return token == "</s>"
        elif position in {1}:
            return token == "0"
        elif position in {2}:
            return token == "1"
        elif position in {4}:
            return token == "3"
        elif position in {5}:
            return token == "<s>"
        elif position in {8, 6}:
            return token == "<pad>"
        elif position in {9}:
            return token == "60000"

    attn_2_1_pattern = select_closest(tokens, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"60000", "</s>", "<s>", "80000"}:
            return k_token == "<pad>"
        elif q_token in {"69000", "62000", "75000"}:
            return k_token == "75000"

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, mlp_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, mlp_1_1_output):
        if token in {"0"}:
            return mlp_1_1_output == 1
        elif token in {"1", "</s>"}:
            return mlp_1_1_output == 11
        elif token in {"2"}:
            return mlp_1_1_output == 3
        elif token in {"3"}:
            return mlp_1_1_output == 4
        elif token in {"69000", "62000", "4"}:
            return mlp_1_1_output == 8
        elif token in {"60000", "75000", "80000"}:
            return mlp_1_1_output == 10
        elif token in {"<s>"}:
            return mlp_1_1_output == 6

    attn_2_3_pattern = select_closest(mlp_1_1_outputs, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"60000", "69000"}:
            return k_token == "60000"
        elif q_token in {"62000", "80000"}:
            return k_token == "80000"
        elif q_token in {"</s>", "75000", "<s>"}:
            return k_token == "<pad>"

    attn_2_4_pattern = select_closest(tokens, tokens, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, mlp_1_0_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(token, position):
        if token in {"0"}:
            return position == 1
        elif token in {"1", "75000"}:
            return position == 2
        elif token in {"69000", "60000", "2", "80000", "3"}:
            return position == 0
        elif token in {"62000", "4", "<s>"}:
            return position == 11
        elif token in {"</s>"}:
            return position == 12

    attn_2_5_pattern = select_closest(positions, tokens, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, mlp_1_0_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_1_output, attn_1_2_output):
        key = (attn_1_1_output, attn_1_2_output)
        if key in {
            ("3", "75000"),
            ("3", "<s>"),
            ("60000", "62000"),
            ("60000", "75000"),
            ("60000", "<s>"),
            ("62000", "60000"),
            ("62000", "62000"),
            ("62000", "69000"),
            ("62000", "75000"),
            ("62000", "80000"),
            ("62000", "<s>"),
            ("69000", "60000"),
            ("69000", "62000"),
            ("69000", "69000"),
            ("69000", "75000"),
            ("69000", "80000"),
            ("69000", "<s>"),
            ("75000", "60000"),
            ("75000", "62000"),
            ("75000", "69000"),
            ("75000", "75000"),
            ("75000", "80000"),
            ("75000", "</s>"),
            ("75000", "<s>"),
            ("80000", "62000"),
            ("80000", "69000"),
            ("80000", "75000"),
            ("80000", "<s>"),
            ("</s>", "</s>"),
            ("</s>", "<s>"),
            ("<s>", "62000"),
            ("<s>", "75000"),
            ("<s>", "<s>"),
        }:
            return 7
        elif key in {
            ("4", "4"),
            ("4", "60000"),
            ("4", "62000"),
            ("4", "69000"),
            ("4", "75000"),
            ("4", "80000"),
            ("4", "<s>"),
            ("60000", "4"),
            ("60000", "60000"),
            ("60000", "69000"),
            ("60000", "80000"),
            ("62000", "4"),
            ("69000", "4"),
            ("75000", "4"),
            ("80000", "4"),
            ("80000", "60000"),
            ("80000", "80000"),
            ("</s>", "4"),
            ("<s>", "4"),
            ("<s>", "60000"),
            ("<s>", "69000"),
            ("<s>", "80000"),
        }:
            return 1
        elif key in {
            ("0", "0"),
            ("0", "60000"),
            ("0", "62000"),
            ("0", "69000"),
            ("0", "75000"),
            ("0", "80000"),
            ("0", "<s>"),
            ("60000", "0"),
            ("62000", "0"),
            ("69000", "0"),
            ("75000", "0"),
            ("80000", "0"),
            ("</s>", "0"),
            ("</s>", "60000"),
            ("</s>", "62000"),
            ("</s>", "69000"),
            ("</s>", "75000"),
            ("</s>", "80000"),
        }:
            return 5
        elif key in {("</s>", "2")}:
            return 10
        return 9

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_1_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(token, attn_2_3_output):
        key = (token, attn_2_3_output)
        if key in {
            ("0", "2"),
            ("0", "4"),
            ("1", "2"),
            ("1", "4"),
            ("2", "2"),
            ("2", "3"),
            ("2", "4"),
            ("2", "60000"),
            ("2", "62000"),
            ("2", "80000"),
            ("3", "2"),
            ("3", "3"),
            ("3", "4"),
            ("3", "60000"),
            ("3", "62000"),
            ("3", "69000"),
            ("3", "80000"),
            ("3", "</s>"),
            ("4", "2"),
            ("4", "4"),
            ("4", "60000"),
            ("4", "69000"),
            ("60000", "2"),
            ("60000", "4"),
            ("62000", "2"),
            ("62000", "4"),
            ("69000", "2"),
            ("69000", "4"),
            ("75000", "2"),
            ("75000", "4"),
            ("80000", "2"),
            ("80000", "4"),
            ("</s>", "2"),
            ("</s>", "4"),
            ("</s>", "60000"),
            ("<s>", "2"),
            ("<s>", "4"),
            ("<s>", "60000"),
        }:
            return 6
        elif key in {("4", "3"), ("4", "</s>")}:
            return 3
        return 0

    mlp_2_1_outputs = [mlp_2_1(k0, k1) for k0, k1 in zip(tokens, attn_2_3_outputs)]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
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
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
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
