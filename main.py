# -*-coding:utf-8-*-

import argparse
import os
import numpy as np
import pandas as pd
import weka.core.jvm as jvm
from io import StringIO
from Bio import SeqIO
from weka.core.converters import Loader
from weka.classifiers import Classifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input file.")
    parser.add_argument("-o", help="Output file.")
    args = parser.parse_args()
    return args.i, args.o


def preprocess(filepath):
    string = ""
    records = SeqIO.parse(filepath, "fasta")
    for record in records:
        for i in range(20, len(record.seq) - 20):
            if record.seq[i].upper() == "A":
                string += ">" + str(record.name) + "_" + str(i + 1) + "\n" + str(record.seq[i - 20:i + 21]) + "\n"
    list = string.split("\n")
    return list[0:-1]


def ohe1(fasta):
    features = ""
    for i in range(164):
        features += "ohe1_f" + str(i + 1) + ","
    features += "class\n"
    nucleotides = ["a", "c", "g", "t"]
    for i in range(0, len(fasta), 2):
        for j in range(len(fasta[i + 1].strip())):
            for k in range(len(nucleotides)):
                if fasta[i + 1][j].lower() == nucleotides[k]:
                    features += "1,"
                else:
                    features += "0,"
        features += "?" + "\n"
    return pd.read_csv(StringIO(features), sep=",")


def ohe2(fasta):
    features = ""
    for i in range(640):
        features += "ohe2_f" + str(i + 1) + ","
    features += "class\n"
    dinucleotides = ["aa", "ac", "ag", "at",
                     "ca", "cc", "cg", "ct",
                     "ga", "gc", "gg", "gt",
                     "ta", "tc", "tg", "tt"]
    for i in range(0, len(fasta), 2):
        for j in range(len(fasta[i + 1].strip()) - 1):
            for k in range(len(dinucleotides)):
                if fasta[i + 1][j:j + 2].lower() == dinucleotides[k]:
                    features += "1,"
                else:
                    features += "0,"
        features += "?" + "\n"
    return pd.read_csv(StringIO(features), sep=",")


def feature_extraction(fasta):
    o1 = ohe1(fasta)
    o2 = ohe2(fasta)
    features = pd.concat([o1.iloc[:, :-1], o2], axis=1)
    selected_indexes = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
                        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                        73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
                        97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
                        117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
                        136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154,
                        155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
                        174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
                        193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 215, 218,
                        234, 250, 312, 314, 333, 344, 352, 353, 355, 356, 357, 358, 359, 370, 381, 382, 388, 397, 404,
                        423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441,
                        442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460,
                        461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
                        480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498,
                        499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517,
                        518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536,
                        537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555,
                        556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574,
                        575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 594, 595, 596, 597, 599, 600,
                        602, 605, 606, 607, 608, 609, 610, 612, 620, 621, 634, 670, 714, 717, 720, 721, 730, 745, 746,
                        747, 748, 749, 750, 751, 752, 804]
    features_selected = features.iloc[:, selected_indexes]
    features_selected.to_csv("temp.csv", index=False)
    data = pd.read_csv("temp.csv")
    columns = data.columns.tolist()
    wf = open("temp.arff", "w+")
    wf.write("@relation temp\n\n")
    for i in range(len(columns) - 1):
        wf.write("@attribute {} ".format(columns[i]) + "numeric\n")
    wf.write("@attribute class {p,n}\n\n")
    wf.write('@data\n')
    with open("temp.csv", 'r') as f:
        f.readline()
        for line in f.readlines():
            wf.write(line)
    wf.close()


def classify():
    jvm.start()
    string = ""
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file("temp.arff", class_index="last")
    classifier, _ = Classifier.deserialize("result.model")
    for index, inst in enumerate(data):
        pred = classifier.classify_instance(inst)
        if pred == 0.0:
            string += "p"
        elif pred == 1.0:
            string += "n"
    jvm.stop()
    return string


def find_last(string, char):
    last_position = -1
    while True:
        position = string.find(char, last_position + 1)
        if position == -1:
            return last_position
        last_position = position


if __name__ == "__main__":
    input_file, output_file = argument_parser()
    dl = preprocess(input_file)
    feature_extraction(dl)
    pred_y = classify()
    result = "No.,Sequence_name,Position,Sequence,Prediction\n"
    for i in range(len(pred_y)):
        sequence = dl[2 * i + 1]
        fl = find_last(dl[2 * i], "_")
        sequence_name = dl[2 * i][0:fl]
        position = dl[2 * i][fl+1:]
        if pred_y[i] == "p":
            prediction = "6mA"
        elif pred_y[i] == "n":
            prediction = "non-6mA"
        result += str(i+1) + "," + sequence_name + "," + position + "," + sequence + "," + prediction + "\n"
    with open(output_file, "w") as file:
        file.write(result)
    os.remove("temp.csv")
    os.remove("temp.arff")
