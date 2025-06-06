import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import floor
from LPF import lowpass, bandpass
from scipy.signal import find_peaks

def find_start_refl(array, len_window=225, mask=0.0435):
    """#+ np.std(array[i:len_window + i], axis=0))"""
    n = len(array)
    max_sum = 0
    max_sum_index = 0
    for i in range(0, n - len_window):
        add_array = np.abs(array[i:i + len_window])
        sum_mask = np.sum(add_array[add_array >= mask], axis=0)
        if sum_mask > max_sum:
            max_sum = sum_mask
            max_sum_index = i
    return max_sum_index

def find_start_refl2(array, len_window=225, mask=0.0435):
    """#+ np.std(array[i:len_window + i], axis=0))"""
    n = len(array)
    max_sum = 0
    max_sum_index = 0
    for i in range(0, n - len_window):
        add_array = np.abs(array[i:i + len_window])
        sum_mask = np.sum(add_array[add_array >= mask], axis=0)
        if sum_mask > max_sum:
            max_sum = sum_mask
            max_sum_index = i
    return max_sum_index




# Обязательные настройки
pulse_freq = 800 # частота стрельбы лазера
ADC_freq = 100*10**6 # частота дискретизации АЦП
file_path = Path(r'D:\WhyNotFreeNames\Work\DAS(ПИШ)\Ф-OTDR check\Test_24.02\das_10V.csv') # где искать файл
col_with_dat_y = 0  # номер столбца с интенсивностями (по порядку начиная с нуля)
len_refl_meters = 1000 # Длина рефлектограммы в метрах
c = 3 * 10**8 # Скорость света
N_g = 1.488 # Коэффициент преломления в волокне
use_mask = 0.2 # Использование маски на данные (0 если не использовать()

# Необязательные параметры
num_refls = None  # число рефлектограмм в одном файле (None если неизвестно)
start_point_in_files = 1000 # обрезание первого количества указанных точек
delete_zero_strings = False # Удаление нулевых строк из каждого получившегося файла. Внимание! Эта настройка может испортить данные, если всего одна рефлектограмма и используются данные с Ф-OTDR!
plt_show_full = True # строить ли графики (НЕ РЕКОМЕНДУЕТСЯ при числе файлов >10)
plt_show_cutted = True
num_cutted_show = 4

# (!) Настроить для лучшей нарезки окон
indent_left = 40 # отступ слева в м для записи в файлы
indent_right = 20  # отступ справа в м для записи в файлы

# Проверка что файл вообще существует
if not file_path.name.endswith('.csv') or file_path.is_dir():
    raise FileNotFoundError(f'File not found: {file_path}')

# Загрузка файла
data = np.loadtxt(file_path, delimiter=';', usecols=tuple(range(col_with_dat_y + 1)),
                  skiprows=1, encoding='utf-8-sig',
                  converters=dict.fromkeys([i for i in range(col_with_dat_y + 1)], lambda x: str(x).replace(',,', '')))

if col_with_dat_y != 0:
    values = data[start_point_in_files:, col_with_dat_y]
else:
    values = data[start_point_in_files:]
N = np.shape(values)[0]
t_file = np.shape(values)[0] / ADC_freq

# Проверка количества рефлектограмм (если это количество указано)
if num_refls is None: #TODO Возможно здесь есть ошибка, но всё сходится
    num_refls = floor(t_file * pulse_freq / 2)
else:
    if num_refls != floor(t_file * pulse_freq / 2):
        raise ValueError("The number of refls in file (", floor(t_file * pulse_freq / 2), ") does not match the written one (", num_refls, ").")

segment_size = int(ADC_freq * (1 / pulse_freq)) # TODO Опять же эта двойка, которой вроде не должно быть

t_refl = (2 * N_g * len_refl_meters) / c
num_refl_points = int(t_refl * ADC_freq)
indent_left_dis = int((2 * N_g * indent_left * ADC_freq) / c)
indent_right_dis = int((2 * N_g * indent_right * ADC_freq) / c)

print("Number of refls in file:", num_refls)
print("Size of one segment:", segment_size)
print("Time of one refl: ", t_refl)
print("The number of discrets in one refls: ", num_refl_points)
print("Total points in file: ", N + start_point_in_files)

indexis = []

for i in range(N - num_refl_points):
    add_array = np.abs(values[i:i + num_refl_points])
    if use_mask:
        indexis.append(np.sum(add_array[add_array >= use_mask], axis=0))
    else:
        indexis.append(np.sum(add_array, axis=0))

indexis = np.array(indexis)
peaks, _ = find_peaks(indexis, distance=segment_size - 0.25 * segment_size, height=np.max(indexis) // 2) # TODO перепроверить datasheet на эту чертову функцию
if plt_show_full:
    plt.plot(indexis)
    plt.plot(peaks, indexis[peaks], 'x')
    plt.show()
elif plt_show_cutted:
    plt.plot(indexis[:num_cutted_show * segment_size])
    plt.plot(peaks[:num_cutted_show], indexis[peaks][:num_cutted_show], 'x')
    plt.show()


# segments = [values[start_point:start_point + int(segment_size) + 1] for start_point in range(start_point_in_files, N, segment_size + 1)] # Нарезка
# if len(peaks) != num_refls:
#     print("Number of segments doesn't match to number of refls. Cutting off the last one.")
#     peaks.pop(-1)
# for segm in segments:
#     plt.plot(bandpass(segm, ADC_freq, 100, 10**6 ))
# plt.show()
# start_points_in_segments = [find_start_refl(segment, num_refl_points) for segment in segments]
# print(start_points_in_segments)
# print(peaks)
refls2 = [values[peaks[i] - indent_left_dis:peaks[i] + num_refl_points + indent_right_dis] for i in range(len(peaks))]
# refls = [segment[start_points_in_segments[i] - indent_left_dis:start_points_in_segments[i] + num_refl_points + indent_right_dis] for i, segment in enumerate(segments)]
# print(refls)
dx = 0.02
if plt_show_full:
    plt.plot([indent_left, indent_left], [np.min(values), np.max(values)], color='black')
    plt.plot([indent_left + num_refl_points, indent_left + num_refl_points], [np.min(values), np.max(values)], color='black')
    for i in range(len(refls2)):
        plt.plot(refls2[i], alpha=0.5)
    plt.show()
elif plt_show_cutted:
    plt.plot([indent_left, indent_left], [np.min(values), np.max(values)], color='black')
    plt.plot([indent_left + num_refl_points, indent_left + num_refl_points], [np.min(values), np.max(values)], color='black')
    for i in range(min(num_cutted_show, len(refls2))):
        plt.plot(refls2[i], alpha=0.5)
    plt.show()
# raise FileNotFoundError()
# mat = np.zeros((np.max([len(refl) for refl in refls]), len(refls)))
# # print(np.shape(mat))
# # print(*list(enumerate(segments)), sep="\n")
# for i, segment in enumerate(segments):
#     if i in [1, 2, 3]:
#         relative_peak_index = np.argmax(segment)
#         relative_peak = np.max(segment)
#         plt.plot(np.linspace(0.01, len(segment), num=len(segment), endpoint=True),
#                  label=f'Segment {i + 1} (Peak: x({relative_peak_index:.5f})={relative_peak:.5f})')
#         plt.plot([relative_peak_index, relative_peak_index], [np.min(segment), np.max(segment)],
#                  color='black', label=f'Segment {i} with max[{relative_peak_index}] = {relative_peak}')
#         plt.plot([indent_left, indent_left], [np.min(segment), np.max(segment)],
#                  color='black', label=f'Argument of segment {i} with index {std_arg[i]}')
#         plt.show()
#     # print(np.shape(mat))
#     mat[0:len(segment), i] = segment
#     print(np.shape(mat), end=" ")
# if delete_zero_strings:
#     mat = mat[~(mat==0).all(1)] # Удаление нулевых строк на всякий случай
# print(np.shape(mat), end=" ")
# pmat = pandas.DataFrame(mat)
# if flag:
#     pass #pmat.to_csv(file_path.absolute().parent /"data_mat" / (file_path.stem + 'mat.csv'), sep=';', header=False, index=False)
# else:
#     print("File " + file_path.stem + " doesn't convert because of too big offset: " + str(np.max(std_arg) - np.min(std_arg)))
#
# if plt_show:
#     plt.xlabel('Point Index')
#     plt.ylabel('Amplitude (с учетом смещения)')
#     plt.legend()
#     plt.title(file_path.name + "\n" + "offset_x = " + str(np.max(std_arg) - np.min(std_arg)))
#     # plt.tight_layout()
#     plt.draw()
#
# print("Количество файлов, где выбрано с помощью сумм:", num_sum)
# print("Количество файлов, где выбрано кросс корреляцией:", num_cross)
# print("Количество файлов, где не выбрано:", num_no_choose)
#
# # print(np.linspace(0.01, segment_size, num=segment_size, endpoint=True))
# if plt_show:
#     plt.show()
