from sklearn import tree
from matplotlib import pyplot as plt


def visualizeTree(tree_model, features, targetname):
    fig = plt.figure(figsize=(60, 60))
    decistion_tree = tree.plot_tree(
        tree_model,
        feature_names=features,
        class_names=targetname,
        filled=True
    )
    # Save picture
    fig.savefig("decistion_tree.png")
