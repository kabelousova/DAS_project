import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

dir_path = Path('./')

for file_path in dir_path.iterdir():
    if not file_path.name.endswith('.csv') or file_path.is_dir():
        continue

    data = pd.read_csv(file_path, delimiter=';', header=None)

    if data[1].dtype == type(str):
        data[1] = data[1].str.replace(',', '.')
        data[1] = pd.to_numeric(data[1], errors='coerce')

    times = data[0].values
    values = data[1].values
    # print(times, values)

    total_points = len(values)
    segment_size = total_points // 4
    segments = [values[i:i + segment_size] for i in range(0, total_points, segment_size)]

    peaks_arg = [np.argmax(segment) for segment in segments]
    segments = [s[max(0, peaks_arg[i] - 195): min(len(s), peaks_arg[i] + 195)] for i, s in enumerate(segments)]

    colors = ['b', 'g', 'r', 'm']
    y_offset = 0.05

    plt.figure(figsize=(10, 8))

    mat = np.zeros([8096, 4])

    for i, segment in enumerate(segments):
        mat[0:len(segment), i] = segment

        relative_peak_index = np.argmax(segment)
        plt.plot(segment + i * y_offset, color=colors[i],
                 label=f'Segment {i + 1} (Peak: {segment[relative_peak_index]:.5f})')

    pmat = pandas.DataFrame(mat)
    pmat.to_csv(file_path.absolute().parent / (file_path.stem + '_mat.csv'), sep=';', header=False)

    plt.xlabel('Point Index')
    plt.ylabel('Amplitude (с учетом смещения)')
    plt.legend()
    plt.title('Наложенные сегменты с вертикальным смещением')
    plt.tight_layout()
    plt.draw()

plt.show()
