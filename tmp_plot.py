import matplotlib.pyplot as plt

f = open("tmp_log")
lines = f.readlines()
f.close()

Y = [ ]

for line in lines:
    if "loss" in line:
        Y.append(line[ line.find("loss") + 6 : line.find("loss") + 12 ])

print(Y)
Y = [float(y) for y in Y]
X = list(range(len(Y)))

plt.plot(X, Y)
plt.show()
