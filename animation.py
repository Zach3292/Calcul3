import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import serial

ser = serial.Serial('dev/cu.usbmodem1101', 115200, timeout=1)

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []


# This function is called periodically from FuncAnimation
def animate(i, xs, ys):
    volts = ser.readline()

    # Add x and y to lists
    xs.append(i)
    ys.append(volts)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Voltage over Time')
    plt.ylabel('Voltage')


# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=1000 / 60, cache_frame_data=False)
plt.show()
