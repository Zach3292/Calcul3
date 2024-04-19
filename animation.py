import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import serial
from scipy import fft as scpfft

ser = serial.Serial('/dev/cu.usbserial-0001', 115200, timeout=None)

# Create figure for plotting
# Parameters
x_len = 100  # Number of points to display
y_range = [0, 5]  # Range of possible Y values to display

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = list(range(0, x_len))
ys = [0] * x_len
ax.set_ylim(y_range)
ax.set_yticks(np.linspace(0, 5, 21))

# Create a blank line. We will update the line in animate
line, = ax.plot(xs, ys)

plt.title('Voltage over Time')
plt.ylabel('Voltage')


# This function is called periodically from FuncAnimation
def animate(i, ys):
    volts = (ser.readline().decode('utf-8').strip())
    if volts == '':
        volts = 0
    else:
        volts = float(volts)
    volts = volts * 5 / 1023

    # Add x and y to lists
    # xs.append(i)
    ys.append(volts)

    # Limit x and y lists to 20 items
    # xs = xs[-20:]
    ys = ys[-x_len:]

    # F = scpfft.fft(ys)

    line.set_ydata(ys)
    # line.set_ydata(np.abs(F))

    return line,


# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(ys,), interval=1, blit=True, cache_frame_data=False)
plt.show()