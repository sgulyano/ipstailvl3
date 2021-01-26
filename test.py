# from sklearn.datasets import load_iris
# from sklearn import tree
# import graphviz
# import matplotlib.pyplot as plt

# iris = load_iris()
# X, y = load_iris(return_X_y=True)
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, y)

# tree.plot_tree(clf)

# dot_data = tree.export_graphviz(clf, out_file='tree.dot', 
#                       feature_names=iris.feature_names,  
#                       class_names=iris.target_names,  
#                       filled=True, rounded=True,  
#                       special_characters=True)  

# fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
# cn=['setosa', 'versicolor', 'virginica']
# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
# tree.plot_tree(clf,
#                feature_names = fn, 
#                class_names=cn,
#                filled = True)
# fig.savefig('imagename.png')

# import matplotlib.pyplot as plt
# import numpy as np
# import plotly.tools as tls

# # redefine some plotting methods for Nextjournal support

# plt.show = plt.gcf

# # Data for plotting
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)

# # Note that using plt.subplots below is equivalent to using
# # fig = plt.figure and then ax = fig.add_subplot(111)
# fig, ax = plt.subplots()
# ax.plot(t, s)

# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#        title='About as simple as it gets, folks')
# ax.grid()

# plt.show()

# tls.mpl_to_plotly(fig)

# import plotly.express as px
from skimage import io
img = io.imread('tree_diagrams/tree_6_2_18.png')
# fig = px.imshow(img)
# fig.show()

import plotly.graph_objects as go
fig = go.Figure(go.Image(z=img))
fig.show()