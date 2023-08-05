import numpy as np
import matplotlib.pyplot as plt
import pywt

# Generate the signal
t = np.linspace(0, 1, 1000)
freq = 10
amp = 5
cos_wave1 = 10 * np.cos(2 * np.pi * 10 * t)
sine_wave2 = amp * np.sin(2 * np.pi * 15 * t)
signal_wave = cos_wave1 + sine_wave2

# wavelet transform
wavelet = 'morl'  # morlet wavelet
scales = np.arange(1, 128)  # Define the scales for the CWT
coefficients, frequencies = pywt.cwt(signal_wave, scales, wavelet)
print(coefficients.shape)
print(coefficients[0])

# Plot the original signal
plt.subplot(2, 1, 1)
plt.plot(t, signal_wave)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original Signal')




# Plot the wavelet coefficients
plt.subplot(2, 1, 2)
plt.imshow(coefficients, extent=[0, 1, 1, 128], cmap='jet', aspect='auto')
plt.xlabel('Time')
plt.ylabel('Scale')
plt.title('Wavelet Transform')
plt.colorbar(label='Magnitude')

plt.tight_layout()
plt.show()
