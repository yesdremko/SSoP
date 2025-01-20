import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import iirfilter, filtfilt

def harmonic_with_noise(t, amplitude, frequency, phase, noise_mean, noise_covariance, show_noise):
    clean_signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if show_noise:
        noise = np.random.normal(noise_mean, np.sqrt(noise_covariance), len(t))
        return clean_signal + noise, clean_signal
    return clean_signal, clean_signal

def apply_filter(signal, filter_order, filter_cutoff, sample_rate):
    b, a = iirfilter(filter_order, filter_cutoff, btype='low', ftype='butter', fs=sample_rate)
    return filtfilt(b, a, signal)

INIT_AMPLITUDE = 1.0
INIT_FREQUENCY = 0.25
INIT_PHASE = 0.0
INIT_NOISE_MEAN = 0.0
INIT_NOISE_COVARIANCE = 0.1
INIT_SHOW_NOISE = True
INIT_FILTER_ORDER = 2
INIT_FILTER_CUTOFF = 0.05
INIT_SHOW_FILTERED = True

t = np.linspace(0, 10, 1000)
sample_rate = 1 / (t[1] - t[0])

fig, ax = plt.subplots()
plt.subplots_adjust(0.09, 0.45, 0.98, 0.97)

signal, clean_signal = harmonic_with_noise(t, INIT_AMPLITUDE, INIT_FREQUENCY, INIT_PHASE, INIT_NOISE_MEAN, INIT_NOISE_COVARIANCE, INIT_SHOW_NOISE)
filtered_signal = apply_filter(signal, INIT_FILTER_ORDER, INIT_FILTER_CUTOFF, sample_rate)

[line_noise] = ax.plot(t, signal, label="Harmonic with Noise", color="orange")
[line_clean] = ax.plot(t, clean_signal, label="Clean Harmonic", color="blue", linestyle="--")
[line_filtered] = ax.plot(t, filtered_signal, label="Filtered Harmonic", color="green", linestyle="-.")
ax.legend()
ax.set_ylim(-2, 2)

ax_amp = plt.axes([0.15, 0.35, 0.54, 0.03])
ax_freq = plt.axes([0.15, 0.30, 0.54, 0.03])
ax_phase = plt.axes([0.15, 0.25, 0.54, 0.03])
ax_noise_mean = plt.axes([0.15, 0.20, 0.54, 0.03])
ax_noise_cov = plt.axes([0.15, 0.15, 0.54, 0.03])
ax_filter_order = plt.axes([0.15, 0.10, 0.54, 0.03])
ax_filter_cutoff = plt.axes([0.15, 0.05, 0.54, 0.03])

slider_amp = Slider(ax_amp, "Amplitude", 0.1, 2.0, valinit=INIT_AMPLITUDE)
slider_freq = Slider(ax_freq, "Frequency", 0.1, 1.0, valinit=INIT_FREQUENCY)
slider_phase = Slider(ax_phase, "Phase", 0.0, 2 * np.pi, valinit=INIT_PHASE)
slider_noise_mean = Slider(ax_noise_mean, "Noise Mean", -1.0, 1.0, valinit=INIT_NOISE_MEAN)
slider_noise_cov = Slider(ax_noise_cov, "Noise Cov", 0.01, 1.0, valinit=INIT_NOISE_COVARIANCE)
slider_filter_order = Slider(ax_filter_order, "Filter Order", 1, 10, valinit=INIT_FILTER_ORDER, valstep=1)
slider_filter_cutoff = Slider(ax_filter_cutoff, "Filter Cutoff", 0.01, 0.5, valinit=INIT_FILTER_CUTOFF)

ax_checkbox_noise = plt.axes([0.79, 0.27, 0.2, 0.1])
checkbox_noise = CheckButtons(ax_checkbox_noise, ["Show Noise"], [INIT_SHOW_NOISE])

ax_checkbox_filtered = plt.axes([0.79, 0.16, 0.2, 0.1])
checkbox_filtered = CheckButtons(ax_checkbox_filtered, ["Show Filtered"], [INIT_SHOW_FILTERED])

ax_reset = plt.axes([0.79, 0.05, 0.20, 0.1])
button_reset = Button(ax_reset, "Reset")

def update(val):
    amplitude = slider_amp.val
    frequency = slider_freq.val
    phase = slider_phase.val
    noise_mean = slider_noise_mean.val
    noise_covariance = slider_noise_cov.val
    filter_order = int(slider_filter_order.val)
    filter_cutoff = slider_filter_cutoff.val
    show_noise = checkbox_noise.get_status()[0]
    show_filtered = checkbox_filtered.get_status()[0]

    signal, clean_signal = harmonic_with_noise(t, amplitude, frequency, phase, noise_mean, noise_covariance, show_noise)
    filtered_signal = apply_filter(signal, filter_order, filter_cutoff, sample_rate)
    
    line_noise.set_ydata(signal)
    line_clean.set_ydata(clean_signal)
    line_filtered.set_ydata(filtered_signal)
    
    line_noise.set_visible(show_noise)
    line_filtered.set_visible(show_filtered)

    fig.canvas.draw_idle()

def reset(event):
    slider_amp.reset()
    slider_freq.reset()
    slider_phase.reset()
    slider_noise_mean.reset()
    slider_noise_cov.reset()
    slider_filter_order.reset()
    slider_filter_cutoff.reset()
    checkbox_noise.set_active(0 if INIT_SHOW_NOISE else 1)
    checkbox_filtered.set_active(0 if INIT_SHOW_FILTERED else 1)

slider_amp.on_changed(update)
slider_freq.on_changed(update)
slider_phase.on_changed(update)
slider_noise_mean.on_changed(update)
slider_noise_cov.on_changed(update)
slider_filter_order.on_changed(update)
slider_filter_cutoff.on_changed(update)
checkbox_noise.on_clicked(update)
checkbox_filtered.on_clicked(update)
button_reset.on_clicked(reset)

plt.show()