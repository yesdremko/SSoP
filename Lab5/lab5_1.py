import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

def harmonic_with_noise(t, amplitude, frequency, phase, noise_mean, noise_covariance, show_noise):
    clean_signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if show_noise:
        noise = np.random.normal(noise_mean, np.sqrt(noise_covariance), len(t))
        return clean_signal + noise, clean_signal
    return clean_signal, clean_signal

INIT_AMPLITUDE = 1.0
INIT_FREQUENCY = 0.25
INIT_PHASE = 0.0
INIT_NOISE_MEAN = 0.0
INIT_NOISE_COVARIANCE = 0.1
INIT_SHOW_NOISE = True


t = np.linspace(0, 10, 1000)

fig, ax = plt.subplots()
plt.subplots_adjust(0.09, 0.35, 0.98, 0.97)

signal, clean_signal = harmonic_with_noise(t, INIT_AMPLITUDE, INIT_FREQUENCY, INIT_PHASE, INIT_NOISE_MEAN, INIT_NOISE_COVARIANCE, INIT_SHOW_NOISE)
[line_noise] = ax.plot(t, signal, label="Harmonic with Noise", color="orange")
[line_clean] = ax.plot(t, clean_signal, label="Clean Harmonic", color="blue", linestyle="--")
ax.legend()
ax.set_ylim(-2, 2)

ax_amp = plt.axes([0.15, 0.25, 0.54, 0.03])
ax_freq = plt.axes([0.15, 0.20, 0.54, 0.03])
ax_phase = plt.axes([0.15, 0.15, 0.54, 0.03])
ax_noise_mean = plt.axes([0.15, 0.10, 0.54, 0.03])
ax_noise_cov = plt.axes([0.15, 0.05, 0.54, 0.03])

slider_amp = Slider(ax_amp, "Amplitude", 0.1, 2.0, valinit=INIT_AMPLITUDE)
slider_freq = Slider(ax_freq, "Frequency", 0.1, 1.0, valinit=INIT_FREQUENCY)
slider_phase = Slider(ax_phase, "Phase", 0.0, 2 * np.pi, valinit=INIT_PHASE)
slider_noise_mean = Slider(ax_noise_mean, "Noise Mean", -1.0, 1.0, valinit=INIT_NOISE_MEAN)
slider_noise_cov = Slider(ax_noise_cov, "Noise Cov", 0.01, 1.0, valinit=INIT_NOISE_COVARIANCE)

ax_checkbox = plt.axes([0.79, 0.17, 0.2, 0.1])
checkbox = CheckButtons(ax_checkbox, ["Show Noise"], [INIT_SHOW_NOISE])

ax_reset = plt.axes([0.79, 0.06, 0.20, 0.1])
button_reset = Button(ax_reset, "Reset")

def update(val):
    amplitude = slider_amp.val
    frequency = slider_freq.val
    phase = slider_phase.val
    noise_mean = slider_noise_mean.val
    noise_covariance = slider_noise_cov.val
    show_noise = checkbox.get_status()[0]

    signal, clean_signal = harmonic_with_noise(t, amplitude, frequency, phase, noise_mean, noise_covariance, show_noise)
    line_noise.set_ydata(signal)
    line_clean.set_ydata(clean_signal)
    fig.canvas.draw_idle()

def reset(event):
    slider_amp.reset()
    slider_freq.reset()
    slider_phase.reset()
    slider_noise_mean.reset()
    slider_noise_cov.reset()
    checkbox.set_active(0 if INIT_SHOW_NOISE else 1)

slider_amp.on_changed(update)
slider_freq.on_changed(update)
slider_phase.on_changed(update)
slider_noise_mean.on_changed(update)
slider_noise_cov.on_changed(update)
checkbox.on_clicked(update)
button_reset.on_clicked(reset)

plt.show()