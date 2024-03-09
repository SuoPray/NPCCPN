import numpy as np
from scipy import io
from sklearn.neighbors import KDTree
from typing import List
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import  accuracy_score
import time

def data_load(data_path: str):
    if data_path.endswith('.csv') or data_path.endswith('.txt'):
        points = np.asarray(np.loadtxt(data_path, delimiter=',', comments='x', usecols=[0, 1]))
    elif data_path.endswith('.mat'):
        data_dict = io.loadmat(data_path)
        # print(data_dict)
        points = data_dict['data']
    else:
        return []
    return points

def plotResult(data, noise_data, SNG):
    scatterColors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'pink', 'indigo'
        , 'tomato', 'teal', 'darksalmon', 'olive', 'darkseagreen', 'thistle']
    # scatterColors = ['black']
    plt.figure()
    for i in range(len(SNG)):
        color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        for j in SNG[i]:
            x1.append(data[j, 0])
            y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, marker='o', alpha=1, s=6)
    x2 = []
    y2 = []
    for i in range(len(noise_data)):
        x2.append(noise_data[i, 0])
        y2.append(noise_data[i, 1])
    plt.scatter(x2, y2, c='black', marker='o', alpha=1, s=6)
    # plt.scatter(x2, y2, c='black', marker='o', alpha=1, s=4)

    plt.title('plotResult')
    plt.show()

def getDensity(data, tree, N):
    """
    计算每个点的密度
    :param data: 数据集
    :param tree: KD树
    :param N: 邻居树
    :return: 密度列表
    """
    # 先计算每个点的N邻域的点的距离
    distance = []
    for i in range(len(data)):
        # 取max(nb_value)的一个弊端，会计算一些无用点的距离，距离与密度的相关性会减弱
        dist, temp = tree.query([data[i]], N + 1)
        # 不包括自己N
        distance.append(dist[0][1:].tolist())
    # 计算每个点N邻域的密度
    density = []
    for i in range(len(distance)):
        if sum(distance[i]) == 0:
            density.append(10000)
        else:
            density.append(N / sum(distance[i]))
    return density

def dataHandle(data, alpha, density, tree):

    clean_data1, noise_data1 = [], []
    clean_data, noise_data = [], []
    length = int(len(density) * alpha / 100)
    noise_list1 = np.argsort(density)[:length]
    clean_list1 = np.argsort(density)[length:]
    for p in noise_list1:
        noise_data1.append(data[p])
    for p in clean_list1:
        clean_data1.append(data[p])

    density_index = []
    # 逆邻居
    index_list = [[] for _ in range(len(data))]
    # 每一个点有多少点指向自己
    nb_values = [0 for _ in range(len(data))]
    # 邻域关系矩阵：哪一个点
    NNr = [[0 for _ in range(len(data))] for _ in range(len(data))]
    # 有别的点指向的点
    clean_data2 = []
    # 没有任何点指向的点
    noise_data2 = []
    # 有指向的点的邻域关系矩阵
    chain_nnr = [[0 for _ in range(len(data))] for _ in range(len(data))]

    for i in range(len(data)):
        r_distance = 0
        for j in range(len(data) - 1):
            r_distance += 1
            temp, index = tree.query([data[i]], r_distance + 1)
            # 计算密度比较
            if density[i] == max(density):
                continue
            elif density[index[0][-1]] <= density[i]:
                continue
            else:
                density_index.append(index[0][-1])
                chain_nnr[i][density_index[-1]] = 1
                if density_index not in index_list[density_index[-1]]:
                    nb_values[density_index[-1]] += 1
                    index_list[density_index[-1]].append(i)
                break;

    for k in range(len(data)):
        if nb_values[k] > 0:
            clean_data2.append(data[k])
        else:
            noise_data2.append(data[k])

    nb_zero_lenth = nb_values.count(0)
    noise_list2 = np.argsort(nb_values)[:nb_zero_lenth]
    clean_list2 = np.argsort(nb_values)[nb_zero_lenth:]

    # 数据规范化
    noise_data2 = np.asarray(noise_data2)
    clean_data2 = np.asarray(clean_data2)

    noise_list = np.union1d(noise_list1, noise_list2)
    clean_list = np.intersect1d(clean_list1, clean_list2)
    for l in range(len(noise_list)):
        noise_data.append(data[noise_list[l]])

    for n in range(len(clean_list)):
        clean_data.append(data[clean_list[n]])

    return np.asarray(chain_nnr),np.asarray(clean_data), np.asarray(noise_data), noise_list, clean_list,np.asarray(clean_data1), np.asarray(noise_data1),np.asarray(clean_data2), np.asarray(noise_data2)

def getAlpha1(var, scale):
    # 设置数据的中位数下标
    standard = scale // 2
    # 只考虑前一半的数据
    var = var[:standard]
    # 首先获取波峰
    max_index = int(np.argmax(var))
    # print(f'波峰:{max_index}')
    # 从波峰后开始寻找稳定的一个值
    # 计算方差图的极差
    var_range = max(var) - var[-1]
    k = standard - max_index
    over = []
    for i in range(1, k + 1):
        # 每次的增益
        step = (var_range / k) * i
        # 每次增益超越的点数
        p = 0
        for j in range(standard - 1, max_index - 1, -1):
            if step > var[j]:
                p += 1
            else:
                break
        over.append(p)
    enhence = []
    enhence.append(over[0])
    for i in range(1, len(over)):
        if over[i] == 0 or over[i - 1] == 0:
            enhence.append(1)
        else:
            enhence.append(over[i] / over[i - 1])
    # print(f'增益:{enhence}')
    # 获得增益最大的那个索引，放到over里面，用50减掉它就是最终的alpha
    a = int(np.argmax(enhence))
    alpha = standard - over[a]
    # 对波峰进行处理，如果波峰出现在中间的话，说明波峰那个点已经是噪声分界点了
    if max_index == 0:
        pass
    else:
        alpha -= 1
    if scale == 100:
        return alpha
    else:
        return alpha * 10

def computeDiff1(density):
    sorted_density = sorted(density)
    diff1 = []
    for i in range(len(density) - 1):
        temp = sorted_density[i + 1] - sorted_density[i]
        diff1.append(temp)
    return diff1

def varCompute(diff1, scale):
    diff1.append(diff1[-1])
    step = len(diff1) // scale
    # print(step)
    var = []
    for i in range(scale):
        start, end = i * step, (i + 1) * step
        temp_diff1 = diff1[start:end]
        temp_var = np.var(temp_diff1)
        var.append(temp_var)
    return var

def NaNN(data, tree):
    # 是否搜索表示位
    flag = 0
    # 搜索了几轮
    r = 0
    # 每个点的邻域信息
    NNr = [[0 for _ in range(len(data))] for _ in range(len(data))]
    # 标记已经被谁当作最近邻
    index_list = [[] for _ in range(len(data))]
    # 每个点的邻域值
    nb_value = [0 for _ in range(len(data))]
    # 计算每一轮邻域值有没有变化
    nb_list = []

    # 进循环标志位
    while flag == 0:
        # 循环NNR即查找每个节点是否都有邻居，若都有邻居则改变标志位结束循环
        for i in range(len(data)):
            if 0 not in NNr[i]:
                if i == len(data) - 1:
                    flag = 1
            else:
                break

        # 若不是每个节点都有邻居则继续循环遍历
        if flag == 0:
            # 从没有邻居的节点开始继续循环
            r += 1
            for i in range(len(data)):
                # temp表示data[i]到最近节点的距离，index表示最近节点在tree的下标
                temp, index = tree.query([data[i]], r + 1)
                # 获取距离信息赋值，index第一个维度的所有元素的最后一个元素值
                min_index = index[0][-1]
                # print(min_index)
                # 第i个节点和min_index之间存在一条边，信息存储在NNR中
                NNr[i][min_index] = 1
                # 判断是否需要将距离i最近节点min_index添加到邻居节点列表中
                if min_index not in index_list[min_index]:
                    nb_value[min_index] += 1
                    index_list[min_index].append(i)
            # 判断若所有节点的邻域值都不为零的个数
            if nb_value.count(0) != 0:
                nb_list.append(nb_value.count(0))
            else:
                # 若都为零则返回结果
                return r, NNr, index_list, nb_value
            # 邻居节点数量最大值列表nb_list中连续两个元素相等，或搜索到表最后一个即nb_list列表的长度等于数据集大小减 1
            if (len(nb_list) >= 2 and nb_list[-1] == nb_list[-2]) or len(nb_list) == len(data) - 1:
                return r, NNr, index_list, nb_value
        # 若每个节点都有邻居则不进入循环
        elif flag == 1:
            return r, NNr, index_list, nb_value

def showNeighborList(nb_value: List[int], index_list: List[List[int]]) -> None:
    print('每个点的邻居值和邻域情况')
    print('index\nnb_value\nnb域')
    for i in range(len(nb_value)):
        print(f'{i}\t\t{nb_value[i]}\t\t{index_list[i]}')
    print('nb的平均值sup：%.2f' % (sum(nb_value) / len(nb_value)))


def NNr_Transforms(NNr):
    # 创建空列表用于存放结果
    NNr_index = []
    # 遍历邻接矩阵NNR的每一行
    for i in range(len(NNr)):
        # 存放当前节点在NNR中的下标
        temp_list = []
        # 遍历第i行的每个节点
        for j in range(len(NNr[i])):
            # 判断是否是邻居节点
            if NNr[i][j] == 1:
                # 将第j个节点在NNR中的下表存入列表
                temp_list.append(j)
        # 将temp_list列表添加到NNR_index列表中，存储当前节点的邻居点在NNR中的下标
        NNr_index.append(temp_list)
    return NNr_index

def showNeighbor(data, NNr_index):
    # 创建一个空白图形对象
    plt.figure()
    # 创建横纵坐标，X表示元素的第一个值，Y表示元素的第二个值
    X, Y = [p[0] for p in data], [p[1] for p in data]
    # 显示数据点颜色，显示样式，透明度，样式大小
    plt.scatter(X, Y, c='b', marker='o', alpha=1, s=6)

    # 根据NNR_index的信息把箭头标注
    for index in range(len(data)):
        # index表示要遍历的节点在NNr_index下的坐标，range生成一个整数序列范围从0到len(NNR_index[index])
        # 遍历NNR_index[index]列表中的每个元素，访问对应节点在原始数据中的下标，可以对每个邻居节点进行操作
        for neighbor in range(len(NNr_index[index])):
            # 散点图添加注释，xy位置信息，数据坐标系，注释文字位置，箭头参数
            plt.annotate('', xy=(data[NNr_index[index][neighbor]][0], data[NNr_index[index][neighbor]][1]),
                         xycoords='data',
                         xytext=(data[index][0], data[index][1]), arrowprops=dict(arrowstyle='->', color='grey',alpha=0.3))
    # 绘图结果显示
    plt.show()

def NoiseConfirm(data, alpha, density):
    length = int(len(density) * alpha / 100)
    noise_list = np.argsort(density)[:length]
    noise_data = []
    for p in noise_list:
        noise_data.append(data[p])
    plt.figure()

    X, Y = [p[0] for p in data], [p[1] for p in data]
    noise_X, noise_Y = [p[0] for p in noise_data], [p[1] for p in noise_data]
    plt.scatter(X, Y, c='b', marker='o', alpha=0.3, s=6)
    plt.scatter(noise_X, noise_Y, c='r', marker='*', alpha=1, s=20)
    # plt.scatter([point[0]], [point[1]], c='r')

    plt.show()

def getSNG(NNr_index, index_list):
    """
    通过近邻信息构建全邻居图
    :param NNr_index: 邻域信息
    :param index_list: 每个点的nb域
    :return: 全邻居图
    """
    SNG = []
    # 每个点的是否访问列表，0代表未访问，1代表已经被访问
    visited = [0 for _ in range(len(NNr_index))]
    while True:
        C = []
        stack = [visited.index(0)]
        while stack:
            p = stack.pop(0)
            visited[p] = 1
            C.append(p)
            # 对p的邻域进行处理
            for q in NNr_index[p]:
                # 若已经是聚过类的跳过不加入
                if visited[q]:
                    pass
                # 没有处理过的就加入队列
                else:
                    stack.append(q)
            # 对p的逆邻居域进行处理
            for q in index_list[p]:
                # 若已经是聚过类的跳过不加入
                if visited[q]:
                    pass
                # 没有处理过的就加入队列
                else:
                    stack.append(q)
            stack = list(set(stack))
        SNG.append(C)
        if visited.count(1) == len(NNr_index):
            break
    # SNG = np.asarray(SNG)
    return SNG

def Redistribution(data, SNG, tree):
    # 先给每一个干净点打上标记
    label = [0] * len(data)
    for p in range(len(SNG)):
        for q in range(len(SNG[p])):
            # SNG[p][q]是clean_data的索引，clean_list[SNG[p][q]]是原始data的索引
            label[clean_list[SNG[p][q]]] = p + 1

    while label.count(0) != 0:
        noise_index = label.index(0)
        find = 1
        while label[noise_index] == 0:
            distance, index = tree.query([data[noise_index]], find + 1)
            compare_point = index[0][-1]
            # 判断该点是不是噪声点，是噪声点就寻找下一个最近的点，若不是，则把噪声点的标签和这个点归一
            if compare_point in noise_list:
                find += 1
                continue
            else:
                label[noise_index] = label[compare_point]

    # 把所有的label进行合并
    # 再对所有的点进行聚类
    clustering = [[] for _ in range(len(SNG))]
    for i in range(len(data)):
        clustering[label[i] - 1].append(i)
    # print(clustering)

    return clustering, label

def plotFinalResult(data, clustering):

    scatterColors = ['red', 'green', 'blue', 'purple', 'brown', 'pink', 'indigo',
                     'tomato', 'teal', 'darksalmon', 'olive', 'darkseagreen', 'thistle']
    plt.figure()

    for i in range(len(clustering)):
        color = scatterColors[i % len(scatterColors)]
        x1 = []
        y1 = []
        for j in clustering[i]:
            x1.append(data[j, 0])
            y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, marker='o', alpha=1, s=6)

    plt.show()

def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def getParameter(clustering, N, data_path, clean_list):

    # 先根据SNG给每个点写出聚类的标签
    label1, real_label = [-1 for _ in range(N)], []
    Nl = 0
    for i in range(len(clustering)):
        Nl += 1
        for p in clustering[i]:
            label1[p] = Nl

    if data_path.split('.')[-1] == 'csv' or data_path.split('.')[-1] == 'txt' or data_path.split('.')[-1] == 'data':
        real_label = list(pd.read_csv(data_path, sep=',', usecols=[2], squeeze=True))

    elif data_path.split('.')[-1] == 'mat':
        data_dict = io.loadmat(data_path)
        label2 = data_dict['label']

        for i in range(len(label2)):
            # 标签单独放在label里
            real_label.append(label2[i][0])

    acc = round(accuracy_score(np.asarray(real_label), np.asarray(label1)), 6)
    purity = round(purity_score(np.asarray(real_label), np.asarray(label1)), 6)
    return  acc, purity


if __name__ == '__main__':
    start_time = time.time()

    data_path = '.\experimentData/data/2sp2glob.csv' #
    # data_path = '.\experimentData/data/3N_3.1_2.csv' #
    # data_path = '.\experimentData/data/banana.csv'  #
    # data_path = '.\experimentData/data/Is.mat' #
    # data_path = '.\experimentData/data/R15.csv' #
    # data_path = '.\experimentData/data/smile3.csv' #
    # data_path = '.\experimentData/data/compound.mat' #
    # data_path = '.\experimentData/data/triangle1.csv' #
    # data_path = '.\experimentData/data/yuan_1500.mat' #
    # data_path = '.\experimentData/data/yueya_2000.mat' #
    # data_path = '.\experimentData/data/cluto-t5-8k.csv'  #
    # data_path = '.\experimentData/data/zelnik4.csv'  #


    data = np.asarray(data_load(data_path))

    print(data.shape)

    if len(data) >= 500:
        scale = 100
    else:
        scale = 10
    # 构建最初原始数据集的KD树
    tree = KDTree(data)

    r, NNr, index_list, nb_value = NaNN(data, tree)

    N = (max(nb_value) + r)

    density = getDensity(data, tree, N)

    diff1 = computeDiff1(density)

    var = varCompute(diff1, scale)

    alpha = getAlpha1(var, scale)

    chain_nnr, clean_data, noise_data, noise_list, clean_list,clean_data1, noise_data1,clean_data2, noise_data2 = dataHandle(data, alpha, density,tree)

    clean_tree = KDTree(clean_data)

    clean_r, clean_NNr, clean_index_list, clean_nb_value = NaNN(clean_data, clean_tree)

    clean_NNr_index = NNr_Transforms(clean_NNr)

    SNG = getSNG(clean_NNr_index, clean_index_list)

    clustering, label = Redistribution(data, SNG, tree)

    plotFinalResult(data, clustering)

    acc, purity = getParameter(clustering, len(data), data_path, clean_list)
    print(f'Accuracy:{acc},Purity:{purity}')
    end_time = time.time()
    print('耗时，%.2f s' % (end_time - start_time))



