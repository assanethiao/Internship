import matplotlib.pyplot as plt

clusters = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

accuracies = [0.3615,0.3795,0.4018,0.4067,0.3734,0.4143,0.3993,0.3963,0.3934,0.3788,0.3724,0.3693,0.3825,0.3791,0.3702]

plt.figure(figsize=(8,5))

plt.plot(clusters, accuracies, marker='o')

plt.xlabel("Number of clusters")
plt.ylabel("Clustering Accuracy")
plt.title("Accuracy vs Number of Clusters")

plt.grid(True)

plt.show()