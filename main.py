import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/kabelousova/PycharmProjects/pythonProject3/test.csv', delimiter=';', header=None)
data[1] = data[1].str.replace(',', '.')
data[1] = pd.to_numeric(data[1], errors='coerce')


times = data[0].values
values = data[1].values
print(times, values)


total_points = len(values)
segment_size = total_points // 4
segments = [values[i:i + segment_size] for i in range(0, total_points, segment_size)]

peaks_arg = [np.argmax(segment) for segment in segments]
segments = [s[max(0, peaks_arg[i] - 195): min(len(s), peaks_arg[i] + 195)] for i, s in enumerate(segments)]


colors = ['b', 'g', 'r', 'm']  
y_offset = 0.05 


plt.figure(figsize=(10, 8))

for i, segment in enumerate(segments):
    relative_peak_index = np.argmax(segment)  
    plt.plot(segment + i * y_offset, color=colors[i], label=f'Segment {i + 1} (Peak: {segment[relative_peak_index]:.5f})')

plt.xlabel('Point Index')
plt.ylabel('Amplitude (с учетом смещения)')
plt.legend()
plt.title('Наложенные сегменты с вертикальным смещением')
plt.tight_layout()
plt.show()

