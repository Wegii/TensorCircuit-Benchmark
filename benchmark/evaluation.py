import matplotlib.pyplot as plt
import csv

batch_size = []
time = []

with open('../profile/tc_execution_cpu.txt', 'r') as datafile:
    plotting = csv.reader(datafile, delimiter=',')

    for row in plotting:
        batch_size.append(row[0])
        time.append(float(row[1]))

plt.plot(batch_size, time, linewidth=2)
plt.title('Execution Time of Quantum Circuit for Different Batch Sizes on CPU')
plt.xlabel('Batch Size')
plt.ylabel('Execution Time [s]')
plt.grid()
ax = plt.gca()
ax.set_ylim([0, max(time)+1])
plt.show()