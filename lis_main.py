import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def cross_cor(segm1, segm2, len_window=10):
    n = np.shape(segm1)[1]
    max_sum = 0
    arr_sum = []
    arg_max = -1
    for i in range(1, n):
        # print(i, n - i)
        # print(segm1[0, 0:i], segm2.T[n - i: n, 0], sep="\n")
        # print(np.shape(segm1))
        # print(np.shape(segm2.T))
        mysum = float((segm1[0, 0:i] * segm2.T[n - i: n, 0])[0, 0])
        if mysum > max_sum:
            max_sum = mysum
            arg_max = i
        # print("sum =", mysum)
        arr_sum.append(mysum)
    # print(8*"=")
    for i in range(0, n):
        # print(i, n - i)
        # print(segm1[0, i:n], segm2.T[i: n, 0], sep="\n")
        mysum = float((segm1[0, i:n] * segm2.T[i: n, 0])[0, 0])
        if mysum > max_sum:
            max_sum = mysum
            arg_max = i
        # print("sum =", mysum)
        arr_sum.append(mysum)

    return np.argmax(arr_sum), max_sum, arr_sum

def find_SKO(array, len_window=225, mask=0.0435):
    """#+ np.std(array[i:len_window + i], axis=0))"""
    n = len(array)
    max_sko = 0
    # max_mean = 0
    # max_std = 0
    max_sko_index = 0
    for i in range(0, n - len_window):
        add_array = array[i:i + len_window]
        mean = np.sum(add_array[add_array >= mask], axis=0)
        std1 = 0 #np.std(add_array, axis=0) * 10**(-10)
        # std = np.mean(array[i:len_window + i], axis=0) * 2 + np.std(array[i:len_window + i], axis=0)
        std = mean + std1
        # print(i, np.mean((array >= mask)[i:len_window + i], axis=0), np.std(array[i:len_window + i], axis=0) * 10)
        # print(i, std, end="")
        if std > max_sko:
            # print(" Added")w
            max_sko = std
            # max_mean = mean
            # max_std = std1
            max_sko_index = i
        # else:
            # print()
    # print(array >= mask)
    # print(max_mean, max_std)
    return max_sko_index

# print(find_SKO(np.array([1, 1,1,1,1,1,2])))
sko_len = 216
plt_show = False
a = 220
b = 5
dir_path = Path('./')
col_with_dat_y = 0
num_refls = 4

segment_size = 2000 # число 2000 кажется оптимальным
num_sum = 0
num_cross = 0
num_no_choose= 0
files_not_converted = []
total_count = 0

for file_path in dir_path.iterdir():
    if not file_path.name.endswith('.csv') or file_path.is_dir():
        continue


    data = pd.read_csv(file_path, delimiter=';', header=None)

    # LIS заменил на 0, т.к. будет только первый столбец
    if data[col_with_dat_y].dtype == type(str):
        data[col_with_dat_y] = data[col_with_dat_y].str.replace(',', '.')
        data[col_with_dat_y] = pd.to_numeric(data[col_with_dat_y], errors='coerce')

    values = data[col_with_dat_y].values
    # times = np.linspace(0.01, len(values), num=len(values), endpoint=True)
    # print(times, values)

    total_points = len(values)
    total_count += 1
    print(str(total_count).rjust(7), " ", end ='', sep='')
    segments = [values[i:i + segment_size] for i in range(0, total_points - 206, segment_size)] # Нарезка

    flag = False

    # Версия с пиками на конце
    # peaks_arg = [np.argmax(segment) for segment in segments]
    # print(peaks_arg, np.max(peaks_arg) - np.min(peaks_arg))
    # if np.max(peaks_arg) - np.min(peaks_arg) <= 0: # Если пики достаточно близки друг к другу, то ориентируемся на них #TODO придумать величину для достаточной близости
    #     segments = [s[max(0, peaks_arg[i] - a): min(len(s), peaks_arg[i] + b)] for i, s in enumerate(segments)]
    #     flag_peaks = True
    #     num_choose_peaks += 1
    std_arg = [find_SKO(segment, sko_len) for segment in segments]
    print("Finding by summing: ", std_arg, np.max(std_arg) - np.min(std_arg))
    if np.max(std_arg) - np.min(std_arg) <= 2:
        flag = True
        segments = [s[max(0, std_arg[i] - b): min(len(s), std_arg[i] + a)] for i, s in enumerate(segments)]
        num_sum += 1
    else:
        cross_corr_arg = [cross_cor(np.matrix(segments[i - 1]), np.matrix(segments[i]))[0] for i in range(1, num_refls)]
        # for i in range(1, num_refls):
        #     print(cross_cor(np.matrix(segments[i - 1]), np.matrix(segments[i]))[0], end=" ")
        # print()
        cross_corr_arg.append(cross_corr_arg[0])
        print(cross_corr_arg)
        if np.max(cross_corr_arg) - np.min(cross_corr_arg) <= 0:
            segments = [s[0: cross_corr_arg[i]] for i, s in enumerate(segments)]
            num_cross += 1
            flag = False
        else:
            num_no_choose += 1
            flag = False
    # for i, s in enumerate(segments):
    #     print(f'Peak {i}: {peaks_arg[i]}')
    #     print(max(0, peaks_arg[i] - 210), min(len(s), peaks_arg[i] + 30))
    #     print(f'{i}: {len(s)}', end='\n')

    if plt_show:
        colors = ['b', 'g', 'r', 'm']
        y_offset = 0.0
        plt.figure(figsize=(16, 12))
        middle_SKO = 0
    mat = np.zeros([np.max([len(segment) for segment in segments]), num_refls])
    # print(np.shape(mat))
    # print(list(enumerate(segments)))
    for i, segment in enumerate(segments):
        # if flag:
        #     pass
        # else: #Обрезка, если ориентировались не по пикам (на глазок)
        #     middle_SKO = find_SKO(segment, sko_len)
        #     peaks_index.append(middle_SKO)
        #     # if np.mean(peaks_arg, axis=0) > 500: # TODO число взято с потолка
        #     #     segment = segment[634:] # TODO число взято с потолка
        #     # else:
        #     #     segment = segment[136:] # TODO число взято с потолка
        #     # segment = segment[:a + b]

        if plt_show:
            relative_peak_index = np.argmax(segment)
            relative_peak = np.max(segment)
            plt.plot(np.linspace(0.01, len(segment), num=len(segment), endpoint=True), segment + i * y_offset,
                     color=colors[i],
                     label=f'Segment {i + 1} (Peak: x({relative_peak_index:.5f})={relative_peak:.5f})')
            plt.plot([relative_peak_index, relative_peak_index], [0, 0.4], color='black')

        mat[0:len(segment), i] = segment
    mat = mat[~(mat==0).all(1)] # Удаление нулевых строк на всякий случай
    pmat = pandas.DataFrame(mat)
    if flag:
        pmat.to_csv(file_path.absolute().parent /"data_mat" / (file_path.stem + 'mat.csv'), sep=';', header=False, index=False)
    else:
        print("File " + file_path.stem + " doesn't convert because of too big offset: " + str(np.max(std_arg) - np.min(std_arg)))

    if plt_show:
        plt.xlabel('Point Index')
        plt.ylabel('Amplitude (с учетом смещения)')
        plt.legend()
        plt.title(file_path.name + "\n" + "offset_x = " + str(np.max(std_arg) - np.min(std_arg)))
        # plt.tight_layout()
        plt.draw()

print("Количество файлов, где выбрано с помощью сумм:", num_sum)
print("Количество файлов, где выбрано кросс корреляцией:", num_cross)
print("Количество файлов, где не выбрано:", num_no_choose)

# print(np.linspace(0.01, segment_size, num=segment_size, endpoint=True))
if plt_show:
    plt.show()
