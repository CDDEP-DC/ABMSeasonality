import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial import ConvexHull

x = np.random.random((100,2)).reshape(100, 2)
y = ((x[:,0]+x[:,1] + np.random.choice([1,-1]) * np.random.random(100)*0.5)>1)*1
plt.scatter(x[:,0],x[:,1],c=y);plt.plot([0,1],[1,0])
model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)
acc = model.score(x, y)
cm = confusion_matrix(y, model.predict(x))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm, alpha= 0.)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Falses', 'Predicted Trues'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual\nFalses', 'Actual\nTrues'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red', fontsize=24)
        ax.tick_params(labelsize=20)

print(classification_report(y, model.predict(x)))
est = model.predict(x)
points = x[est==True]
hull = ConvexHull(points)
x_hull = np.append(points[hull.vertices,0],points[hull.vertices,0][0])
y_hull = np.append(points[hull.vertices,1],points[hull.vertices,1][0])
plt.fill(x_hull, y_hull, alpha=0.3, c='grey')
plt.scatter(x[:,0],x[:,1],c=y)
