import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/kabelousova/PycharmProjects/pythonProject3/test9.csv', delimiter=';', header=None)

data[1] = data[1].str.replace(',', '.')
data[1] = pd.to_numeric(data[1], errors='coerce')

times = data[0].values
values = data[1].values

total_points = len(values)

segment_size = total_points // 4

segments = [values[i:i + segment_size] for i in range(0, total_points, segment_size)]

peaks = [np.max(segment) for segment in segments]

fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

for i, segment in enumerate(segments):
    axs[i].plot(segment)
    axs[i].set_title(f'Segment {i+1} (Peak: {peaks[i]:.5f})')
    axs[i].set_ylabel('Amplitude')

plt.xlabel('Point Index')
plt.tight_layout()
plt.show()