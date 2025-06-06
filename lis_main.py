import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from LPF import lowpass, bandpass
from scipy.signal import correlate, correlation_lags

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

def find_SKO(array, len_window=225, mask=0.030):
    """#+ np.std(array[i:len_window + i], axis=0))"""
    n = len(array)
    max_sko = 0
    max_sko_index = 0
    for i in range(0, n - len_window):
        add_array = np.abs(array[i:i + len_window])
        sum_mask = np.sum(add_array[add_array >= mask], axis=0)
        if sum_mask > max_sko:
            max_sko = sum_mask
            max_sko_index = i
    return max_sko_index

# print(find_SKO(np.array([1, 1,1,1,1,1,2])))

sko_len = 220  # Примерная длина рефлектограммы (в отсчётах)
plt_show = True # строить ли графики (НЕ РЕКОМЕНДУЕТСЯ при числе файлов >10)
plt_err_show = True

max_offset = 2

# (!) Настроить для лучшей нарезки окон
indent_left = 5 # отступ слева
indent_right = sko_len + 5  # отступ справа

dir_path = Path(r'D:\WhyNotFreeNames\Work\DAS(ПИШ)\Камчатка_обработка_Белоусова\Характерные файлы') # где искать файлы
# dir_path = Path(r'D:\WhyNotFreeNames\Work\DAS(ПИШ)\Камчатка_обработка_Белоусова\data_1')
col_with_dat_y = 0  # номер столбца с интенсивностями (по порядку начиная с нуля)
num_refls = 4  # число рефлектограмм в одном файле

delete_zero_strings = True # Удаление нулевых строк из каждого получившегося файла.
# Внимание! Эта настройка может испортить данные, если всего одна рефлектограмма и используются данные с Ф-OTDR!

segment_size = 2000  # количество точек в нарезаемых сегментах

# Просто подсчёт, что удалось скомпилировать, а что нет
num_sum = 0
num_cross = 0
num_no_choose= 0
files_not_converted = []
# lst = [str(i) for i in range(2 + 1)]
# dict_of_offset = dict()
# for i in range(max_offset + 1):
#     dict_of_offset[str(i)] = 0

total_count = 0

if plt_show:
    colors = ['b', 'g', 'r', 'm']
    y_offset = 0.0
    plt.figure(figsize=(16, 12))
    middle_SKO = 0

for file_path in dir_path.iterdir():
    if total_count >= 10000:
        break
    if not file_path.name.endswith('.csv') or file_path.is_dir():
        continue
    if os.path.exists(r'D:\WhyNotFreeNames\Work\DAS(ПИШ)\Камчатка_обработанные_данные\data_mat_ES_30.09.2024_23.38.46_csv_data_ver2' + '\\' + file_path.name):
        print(str(total_count).rjust(7), file_path.name, "was skipped because of exists.")
        total_count += 1
        continue

    data = pd.read_csv(file_path, delimiter=';', header=None)

    if data[col_with_dat_y].dtype == type(str):
        data[col_with_dat_y] = data[col_with_dat_y].str.replace(',', '.')
        data[col_with_dat_y] = pd.to_numeric(data[col_with_dat_y], errors='coerce')

    values = bandpass(data[col_with_dat_y].values, 10**8, 10, 10**7 - 1)
    values -= values.mean(axis=0)
    # print("Mean: ", str(round(values.mean(axis=0), 5)).ljust(8), end='')
    # times = np.linspace(0.01, len(values), num=len(values), endpoint=True)
    # print(times, values)

    total_points = len(values)
    total_count += 1
    print(str(total_count).rjust(7), " ", end='', sep='')
    segments = [values[start_point:start_point + segment_size] for start_point in range(0, num_refls * segment_size, segment_size)] # Нарезка

    cross = [correlate(segments[0], segments[i]) for i in range(1, len(segments))]
    lags = [correlation_lags(len(segments[0]), len(segments[i])) for i in range(1, len(segments))]
    for i, array in enumerate(cross):
        plt.xlabel('Lag')
        plt.ylabel('Cross correlation number')
        plt.title(file_path.name + " segments=" + str(0) + ", " + str(i+1) + " lag = " + str(np.argmax(array) - len(segments[0])))
        plt.plot(lags[i], array)
        plt.show()

    flag = False

    std_arg = [find_SKO(segment, sko_len) for segment in segments]
    print("Finding by summing: ", std_arg, np.max(std_arg) - np.min(std_arg), "Max_sko_index =", std_arg[0], end=" ")
    if std_arg[0] != 0 and np.max(std_arg) - np.min(std_arg) <= max_offset:  #старый вариант
        flag = True
        # dict_of_offset[str(np.max(std_arg) - np.min(std_arg))] += 1
        segments = [s[std_arg[i] - indent_left: std_arg[i] + indent_right] for i, s in enumerate(segments)]
        num_sum += 1
    else:
        print("\nFile did not converted:", file_path.name)
        if plt_err_show:
            plt.plot(np.linspace(0, len(values), num=len(values), endpoint=True), values)
            mat = np.zeros((np.max([len(segment) for segment in segments]), num_refls))
            for i, segment in enumerate(segments):
                if plt_show:
                    relative_peak_index = np.argmax(segment)
                    relative_peak = np.max(segment)

                    plt.plot(np.linspace(0, len(segment), num=len(segment), endpoint=True), segment + i * y_offset,
                             color=colors[i],
                             label=f'Segment {i + 1} (Peak: x({relative_peak_index:.5f})={relative_peak:.5f})')
                mat[0:len(segment), i] = segment
            plt.plot(np.linspace(0, len(mat.mean(axis=1)), num=len(mat.mean(axis=1)), endpoint=True), mat.mean(axis=1), linewidth=3,
                     color='black')
            plt.xlabel('Point Index')
            plt.ylabel('Amplitude (с учетом смещения)')
            # plt.legend()
            plt.title(file_path.name + "\n" + "offset_x = " + str(np.max(std_arg) - np.min(std_arg)))
            # plt.tight_layout()
            plt.show()
        raise ValueError(' ')


    mat = np.zeros((np.max([len(segment) for segment in segments]), num_refls))
    # print(np.shape(mat))
    # print(*list(enumerate(segments)), sep="\n")

    for i, segment in enumerate(segments):
        # print(np.shape(mat))
        mat[0:len(segment), i] = segment
        # print(np.shape(mat), end=" ")
    if delete_zero_strings:
        mat = mat[~(mat==0).all(1)] # Удаление нулевых строк на всякий случай
    print("Total size of mat: ", np.shape(mat), end=' ')
    print(file_path.name)
    one_array = mat.mean(axis=1)
    one_array_pd = pd.DataFrame(one_array)
    pmat = pd.DataFrame(mat)
    if flag:
        pass
        # one_array_pd.to_csv(r'D:\WhyNotFreeNames\Work\DAS(ПИШ)\Камчатка_обработанные_данные\data_mat_ES_30.09.2024_23.38.46_csv_data_ver2' + '\\' + file_path.name, sep=';', header=False, index=False)
        # one_array_pd.to_csv(r'D:\WhyNotFreeNames\Work\DAS(ПИШ)\Камчатка_обработка_Белоусова\data_mat' + '\\' + file_path.name, sep=';', header=False, index=False)
    else:
        print("File " + file_path.stem + " doesn't convert because of too big offset: " + str(np.max(std_arg) - np.min(std_arg)))

    if plt_show:
        for i, segment in enumerate(segments):
            relative_peak_index = np.argmax(segment)
            relative_peak = np.max(segment)

            plt.plot(np.linspace(0, len(segment), num=len(segment), endpoint=True), segment + i * y_offset,
                     color=colors[i],
                     label=f'Segment {i + 1} (Peak: x({relative_peak_index:.5f})={relative_peak:.5f})')
        plt.plot(np.linspace(0, len(one_array), num=len(one_array), endpoint=True), one_array, linewidth=3, color='black')
        plt.xlabel('Point Index')
        plt.ylabel('Amplitude')
        # plt.legend()
        plt.title(file_path.name + "\n" + "offset_x = " + str(np.max(std_arg) - np.min(std_arg)))
        # plt.tight_layout()
        plt.show()

# print("Dictionary of offsets:")
# for i in range(max_offset + 1):
#     print("Offset = ", str(i), ", Count = ", dict_of_offset[str(i)], sep='')
print("Количество файлов, где выбрано с помощью сумм:", num_sum)
print("Количество файлов, где выбрано кросс корреляцией:", num_cross)
print("Количество файлов, где не выбрано:", num_no_choose)

# print(np.linspace(0.01, segment_size, num=segment_size, endpoint=True))
if plt_show:
    plt.show()
