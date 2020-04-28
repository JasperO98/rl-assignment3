import matplotlib.pyplot as plt
from statistics import mean


def opruim():
    new_readlines = []
    unique_avg = []
    unique_max = []
    unique_min = []
    stats = open("stats_model.txt", 'r')
    lines = stats.readlines()
    stats.close()
    for line in lines:
        splito = line.split("\t")
        if splito[0] == "AVG" and splito[1] not in unique_avg: new_readlines.append(line); unique_avg.append(splito[1])
        if splito[0] == "MAX" and splito[1] not in unique_max: new_readlines.append(line); unique_max.append(splito[1])
        if splito[0] == "MIN" and splito[1] not in unique_min: new_readlines.append(line); unique_min.append(splito[1])
    return new_readlines


def get_data():
    new_readlines = opruim()
    avg_scores = {}
    min_scores = {}
    max_scores = {}
    for stat in new_readlines:
        split_stat = stat.split("\t")
        try:
            if split_stat[0] == "AVG": avg_scores[split_stat[1][:-3]].append(float(split_stat[2].rstrip("\n")))
            if split_stat[0] == "MAX": max_scores[split_stat[1][:-3]].append(float(split_stat[2].rstrip("\n")))
            if split_stat[0] == "MIN": min_scores[split_stat[1][:-3]].append(float(split_stat[2].rstrip("\n")))
        except KeyError:
            if split_stat[0] == "AVG": avg_scores[split_stat[1][:-3]] = [float(split_stat[2].rstrip("\n"))]
            if split_stat[0] == "MAX": max_scores[split_stat[1][:-3]] = [float(split_stat[2].rstrip("\n"))]
            if split_stat[0] == "MIN": min_scores[split_stat[1][:-3]] = [float(split_stat[2].rstrip("\n"))]
    return avg_scores, min_scores, max_scores


def get_top_n(dict1, dict2, n=10):
    top_n = sorted(dict2, key=(lambda key: dict2[key]), reverse=True)[0:n]
    newdict = {k: dict1[k] for k in top_n}
    return newdict


def get_tail_n(dict1, dict2, n=10):
    top_n = sorted(dict2, key=(lambda key: dict2[key]), reverse=False)[0:n]
    newdict = {k: dict1[k] for k in top_n}
    return newdict


def get_average_of_lists(model):
    return {k: mean(v) for k, v in model.items()}


def task2_boxplots(dicto, file):
    values = list(dicto.values())[::-1]
    keys = list(dicto.keys())[::-1]
    plt.boxplot(values,
                vert=0,
                patch_artist=True,
                labels=keys)
    plt.savefig("pdf/" + file, bbox_inches='tight')
    plt.show()
    plt.close()


def rename_key(dicto):
    new_dict = {}
    for key in dicto.copy().keys():
        split_keys = key.split("_")
        new_label = "B" + split_keys[0] + " G" + split_keys[1] + " A" + split_keys[2] + " W" + split_keys[3]
        new_dict[new_label] = dicto.pop(key)
    return new_dict


if __name__ == '__main__':
    avg_scores, min_scores, max_scores = get_data()
    for a, b in avg_scores.items():
        if len(b) != 4: print(a, len(b))
    list_of_all_averages = avg_scores.values()
    average_AVG = get_average_of_lists(avg_scores)

    AVG_top10 = get_top_n(avg_scores, average_AVG, 15)
    print("Top\n"
          "model\tvalues\tavarage values")
    for i in AVG_top10:
        print(i + "\t" + str(avg_scores[i]) + "\t " + str(average_AVG[i]))
    AVG_top10 = rename_key(AVG_top10)
    task2_boxplots(AVG_top10, "max1.pdf")

    AVG_tail10 = get_tail_n(avg_scores, average_AVG, 15)
    print("\nTail\nmodel\tvalues\taverage values")
    for i in AVG_tail10:
        print(i + "\t" + str(avg_scores[i]) + "\t " + str(average_AVG[i]))
    AVG_tail10 = rename_key(AVG_tail10)
    task2_boxplots(AVG_tail10, "min1.pdf")

    AVG_tail10 = get_tail_n(avg_scores, average_AVG, 999)
    AVG_tail10 = rename_key(AVG_tail10)
    task2_boxplots(AVG_tail10, "all1.pdf")
