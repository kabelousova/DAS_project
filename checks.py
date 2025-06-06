import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from LPF import lowpass, bandpass
from scipy.signal import find_peaks
from scipy.signal import periodogram
from scipy.fft import fft, fftfreq, rfft, rfftfreq

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

sko_len = 220  # Примерная длина рефлектограммы (в отсчётах)
plt_show = False # строить ли графики (НЕ РЕКОМЕНДУЕТСЯ при числе файлов >10)
plt_err_show = True
len_window_sin = 1000 # Длина окна с синусом в отсчётах

max_offset = 2

# (!) Настроить для лучшей нарезки окон
indent_left = 5 # отступ слева
indent_right = sko_len + 5  # отступ справа

dir_path = Path(r'E:\DASData_2024-09-23_05-05-52.507555__2024-10-27_18-06-00.916381\2024-10-01_11-34-09_2') # где искать файлы
# dir_path = Path(r'D:\WhyNotFreeNames\Work\DAS(ПИШ)\Камчатка_обработка_Белоусова\data_1')
col_with_dat_y = 0  # номер столбца с интенсивностями (по порядку начиная с нуля)
num_refls = 4  # число рефлектограмм в одном файле

use_mask = 0.04

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

file_path = Path(r'E:\DASData_2024-09-23_05-05-52.507555__2024-10-27_18-06-00.916381\2024-10-01_11-34-09_2\DASdata_33480001_2024-10-01_11-34-09.568_100000000Hz_8192_.csv')
total_count = 0

if plt_show:
    colors = ['b', 'g', 'r', 'm']
    y_offset = 0.0
    plt.figure(figsize=(16, 12))
    middle_SKO = 0

data = pd.read_csv(file_path, delimiter=';', header=None)

if data[col_with_dat_y].dtype == type(str):
    data[col_with_dat_y] = data[col_with_dat_y].str.replace(',', '.')
    data[col_with_dat_y] = pd.to_numeric(data[col_with_dat_y], errors='coerce')

indexis = list()
values = data[col_with_dat_y].values
for i in range(len(values) - sko_len):
    add_array = np.abs(values[i:i + sko_len])
    if use_mask:
        indexis.append(np.sum(add_array[add_array >= use_mask], axis=0))
    else:
        indexis.append(np.sum(add_array, axis=0))

plt.plot(values)
plt.show()

indexis = np.array(indexis)
peaks, _ = find_peaks(indexis, distance=segment_size - 0.25 * segment_size, height=np.max(indexis) // 2) # TODO перепроверить datasheet на эту чертову функцию
plt.plot(indexis)
plt.plot(peaks, indexis[peaks], 'x')
plt.show()

flag = False
i = 1
while not flag:
    if len(list(filter(lambda x: x, map(lambda x: False if x == 0 else True, indexis[i:i + len_window_sin])))) == 0:
        flag = True
    else:
        i += 1
if flag:
    sin_arr = values[i + 20:i + 20 + len_window_sin] #TODO магическое число 20
    print(i)
else:
    raise ValueError('Синусоида не найдена')
bp_sin_arr = bandpass(sin_arr, 10**6,1, 10**4)
np.savetxt('Wave_sin_bpf.gz', bp_sin_arr, delimiter='\n')
np.savetxt('Wave_sin.gz', sin_arr, delimiter='\n')
N = len(bp_sin_arr)
T = 10**(-8)
phase = 0  #-1.2 * np.pi
x = np.linspace(phase, N * T + phase, num=N, endpoint=False)
plt.plot(x, bp_sin_arr / np.max(bp_sin_arr))
# y = np.sin(4*10**5 * 2.0*np.pi*x)
# plt.plot(x, y / np.max(np.abs(y)))
plt.show()

freqs, spectrum = periodogram(bp_sin_arr, fs=len(x)/(max(x)-min(x)))

plt.plot(freqs, spectrum)
plt.show()
freqs_inds = np.argmax(spectrum)
fft_amps = np.abs(spectrum)
fft_phase = (np.angle(spectrum[freqs_inds]) - min(x) * spectrum[freqs_inds]) / np.pi % 2
factor = freqs[freqs_inds]
amps = fft_amps[freqs_inds] * 2 / len(x)
print("Frequency indexis:", freqs_inds)
print("Frequency:", factor)
print("Phase:", fft_phase)
print("Amplitudes:", amps)
# print(freqs_inds)


#FFT
# sin_arr = lowpass(sin_arr, 10**4, 10**6)

yf = fft(bp_sin_arr)
xf = fftfreq(N, T)[:N // 2]
plt.plot(xf, np.abs(yf[0:N // 2]))
plt.show()

sin_freq = xf[np.argmax(np.abs(yf))]
# print(sin_freq)

# freqs, spectrum = periodogram(y, fs=len(x)/(max(x)-min(x)))
# plt.plot(freqs, spectrum)
