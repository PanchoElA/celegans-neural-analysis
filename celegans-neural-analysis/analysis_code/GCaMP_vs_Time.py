import pynwb
import matplotlib.pyplot as plt
import numpy as np

### SCRIPT FROM NEUROSIFT ###

f = 'sub-2023-01-09-15-SWF702_ses-20230109_behavior+image+ophys.nwb'
nwb = pynwb.NWBHDF5IO(f, mode='r').read()

CalciumImageSeries = nwb.acquisition["CalciumImageSeries"]  # (MultiChannelVolumeSeries) GCaMP6s series images.
# Dimensions should be (t, x, y, z, C).
# CalciumImageSeries.starting_time 0 sec
# CalciumImageSeries.rate  1.7 Hz

Behavior = nwb.processing["Behavior"]  # (ProcessingModule) Behavioral data

velocity = nwb.processing["Behavior"]["velocity"]  # (BehavioralTimeSeries)

CalciumActivity = nwb.processing["CalciumActivity"]  # (ProcessingModule) Calcium time series metadata, segmentation,
# and fluorescence data

NeuronIDs = nwb.processing["CalciumActivity"]["NeuronIDs"]  # (SegmentationLabels)

SignalRawFluor = nwb.processing["CalciumActivity"]["SignalRawFluor"]  # (Fluorescence)

SignalCalciumImResponseSeries = nwb.processing["CalciumActivity"]["SignalRawFluor"]["SignalCalciumImResponseSeries"]
# (RoiResponseSeries) Raw calcium fluorescence activity


##### Finding Neurons That I Want ######

neuron_names = NeuronIDs.labels[:]
neurons_of_interest = ["RIMR", "AIBR", "AVAR"]
neural_data = np.array(SignalCalciumImResponseSeries.data[:])
for i in range(len(neuron_names)):
    if neuron_names[i] == neurons_of_interest[0]:
        y1n = neuron_names[i]
        id1 = i
    if neuron_names[i] == neurons_of_interest[1]:
        y2n = neuron_names[i]
        id2 = i
    if neuron_names[i] == neurons_of_interest[2]:
        y3n = neuron_names[i]
        id3 = i

##### Create Plots ######

time = SignalCalciumImResponseSeries.timestamps[:]
timestamps = (time - time[0]) / 60
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6))

y1 = neural_data[:, id1]
y2 = neural_data[:, id2]
y3 = neural_data[:, id3]

# Plot data on each subplot
ax1.plot(timestamps, y1, color='blue')
ax1.set_ylabel(y1n)
ax1.set_title('Neuron Activity (lumens) Vs. Time (min)')  # Title for the entire figure, or just the top plot

ax2.plot(timestamps, y2, color='green')
ax2.set_ylabel(y2n)

ax3.plot(timestamps, y3, color='red')
ax3.set_xlabel('Time (min)')
ax3.set_ylabel(y3n)

# Adjust layout to prevent labels from overlapping
plt.tight_layout()

### MOSTRAR DISTINTOS COMPORTAMIENTOS ###

v_data = Behavior["velocity"]["velocity"].data[:]

intervals = np.zeros(len(v_data))
for i in range(len(v_data)):
    if v_data[i] < 0:
        intervals[i] = -1
    elif v_data[i] > 0:
        intervals[i] = 1
    else:
        intervals[i] = 0

neg = 0
pos = 0
count = 0
list_change = [0]
for i in intervals:
    count += 1
    if i > 0:
        pos += 1
    elif i < 0:
        neg += 1
    if count < 1615:
        if i != intervals[count]:
            list_change.append(count)
    else:
        next

count = 0
for i in range(1, len(list_change), 2):
    ax1.axvspan(timestamps[list_change[i - 1]], timestamps[list_change[i]], facecolor='green', alpha=0.3,
                label='Highlighted Region')
    ax2.axvspan(timestamps[list_change[i - 1]], timestamps[list_change[i]], facecolor='green', alpha=0.3,
                label='Highlighted Region')
    ax3.axvspan(timestamps[list_change[i - 1]], timestamps[list_change[i]], facecolor='green', alpha=0.3,
                label='Highlighted Region')

### Display the plot ###
plt.show()
