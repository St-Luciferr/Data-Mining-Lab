import numpy as np
import matplotlib.pyplot as plt

#generate signal
t=np.linspace(0,1,1000)
freq =10
amp=5
cos_wave1=10*np.cos(2*np.pi*10*t)
sine_wave2=amp*np.sin(2*np.pi*15*t)
sine_wave=cos_wave1+sine_wave2

#fourrier transform
f_transform=np.fft.fft(sine_wave)
freqs=np.fft.fftfreq(len(sine_wave),1/len(t))

#inverse fourier transform
inverse_fft=np.fft.ifft(f_transform)

# Plot the original signal
plt.subplot(3, 1, 1)
plt.plot(t, sine_wave)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Time Signal')
plt.xticks(np.arange(0, 1, 0.1))
plt.grid(True)  

# Plot the Fourier transform
plt.subplot(3, 1, 2)
plt.plot(freqs, np.abs(f_transform))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Fourier Transform')
plt.xlim(-30,30)
plt.xticks(np.arange(-30, 30, 5))
plt.grid(True)  

# Plot the Inverse Fourier transform
plt.subplot(3, 1, 3)
plt.plot(t, inverse_fft)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Inverse Fourier Transform')
plt.grid(True)  

plt.tight_layout()
plt.show()