import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
from random import choices
from scipy.spatial.distance import cdist
from scipy.stats import chi2
import matplotlib as mpl

# mpl.rcParams['axes.titlesize'] = 'large'

sns.set_theme()


def remove_outliers_multivariate(data, significance=0.3):
    # Calculate mean and covariance matrix
    mean = np.mean(data, axis=0)
    cov = np.cov(data.T)

    # Calculate the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov)

    # Calculate the Mahalanobis distance for each data point
    distances = []
    for point in data:
        diff = point - mean
        distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
        distances.append(distance)

    # Set the threshold based on the chi-square distribution
    threshold = chi2.ppf(1 - significance, df=len(data[0]))

    # Identify outliers
    outliers = np.where(np.array(distances) > threshold)[0]

    # Remove outliers from the data
    filtered_data = np.delete(data, outliers, axis=0)

    return filtered_data


def get_embeddings(mean=None, cov=None, samples_count=100):
    # np.random.seed(seed)
    # Parameters of the Gaussian distribution
    # mean = [0, 0]  # Mean of the distribution
    # cov = [[1, 0], [0, 1]]  # Covariance matrix of the distribution

    if cov is None:
        cov = [[1, 0], [0, 1]]
    if mean is None:
        mean = [0, 0]
    noise = np.random.multivariate_normal(mean, cov, samples_count)

    # Generate sample EEG embeddings for source and target domains

    source_embeddings = np.random.randn(samples_count, 2) * noise  # Sample embeddings for the source domain
    target_embeddings = np.random.randn(samples_count, 2) * noise + 2  # Sample embeddings for the target domain

    source_embeddings = remove_outliers_multivariate(source_embeddings)
    target_embeddings = remove_outliers_multivariate(target_embeddings)

    # Concatenate the embeddings and create corresponding labels
    X = np.concatenate((source_embeddings, target_embeddings))
    y = np.concatenate((np.zeros(source_embeddings.shape[0]), np.ones(target_embeddings.shape[0])))

    # Train the SVM classifier
    clf = svm.SVC(kernel='linear')
    clf.fit(X, y)

    # Get the coefficients of the decision boundary line
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), samples_count)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    return source_embeddings, target_embeddings, xx, yy


def plot(ax, source_embeddings, target_embeddings, xx, yy,
         title="title", marker="o", marker2=None, class_1="A",
         class_2="B", size=120, color_1='darkblue', color_2='darkorange'):
    # styles = {class_1: marker, class_2: marker}
    # group = [class_1] * source_embeddings.shape[0]
    # group.extend([class_2] * target_embeddings.shape[0])
    # group2 = choices([class_1, class_2], k=source_embeddings.shape[0] + target_embeddings.shape[0])
    # group = np.array(group)
    # sns.scatterplot(ax=ax, x=np.concatenate((source_embeddings[:, 0],target_embeddings[:, 0]), axis=0),
    #                 y=np.concatenate((source_embeddings[:, 1],target_embeddings[:, 1]), axis=0),
    #                 style=group if marker2 is None else group2,
    #                 markers=styles,
    #                 hue=group,
    #                 )

    sns.scatterplot(ax=ax)

    if marker2 is not None:
        half = source_embeddings.shape[0] // 2
        ax.scatter(source_embeddings[:half, 0], source_embeddings[:half, 1],
                   marker=marker,
                   color=color_1,
                   s=size,
                   label=class_1)
        ax.scatter(source_embeddings[half:, 0], source_embeddings[half:, 1],
                   marker=marker2,
                   color=color_1,
                   s=size,
                   label=class_1)
        half = target_embeddings.shape[0] // 2
        ax.scatter(target_embeddings[:half, 0], target_embeddings[:half, 1],
                   marker=marker,
                   color=color_2,
                   s=size,
                   label=class_2)
        ax.scatter(target_embeddings[half:, 0], target_embeddings[half:, 1],
                   marker=marker2,
                   color=color_2,
                   s=size,
                   label=class_2)
    else:
        ax.scatter(source_embeddings[:, 0], source_embeddings[:, 1],
                   color=color_1,
                   marker=marker,
                   s=size,
                   label=class_1

                   )
        ax.scatter(target_embeddings[:, 0], target_embeddings[:, 1],
                   color=color_2,
                   marker=marker,
                   s=size,
                   label=class_2
                   )

    # Plot the decision boundary
    ax.plot(xx, yy, linestyle='dashed', color='black',
            #  label='Decision Boundary'
            )
    title = ax.set_title(title,pad=40)
    title.set_fontsize(24)
    title.set_fontweight("bold")

    return ax


def run():
    source_embeddings, target_embeddings, xx, yy = get_embeddings(mean=[4, 0], cov=[[2, 0.5], [0.5, 1]])
    source_embeddings1, target_embeddings1, xx1, yy1 = get_embeddings(mean=[6, 0], cov=[[2, 0], [0, 0.5]])
    source_embeddings2, target_embeddings2, xx2, yy2 = get_embeddings(mean=[3, 0], cov=[[1, 0.8], [0.8, 1]])
    # Plotting the EEG embeddings
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    for ax in axes.flat:
        # ax.axis('off')
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)
        pass
    color_1 = np.array([91, 126, 182]) / 255

    color_2 = np.array([222, 150, 112]) / 255
    ax = plot(axes[0], source_embeddings, target_embeddings, xx, yy, title="Subject 1", marker="x",
              color_1=color_1,
              color_2=color_2)
    plot(axes[1], source_embeddings1, target_embeddings1, xx1, yy1, title="Subject 2",
         marker='o', color_1=color_1,
         color_2=color_2)
    plot(axes[2], source_embeddings2, target_embeddings2, xx2, yy2, title="Shared Embedding Space", marker="x",
         color_1=color_1, color_2=color_2,
         marker2="o")
    handles, labels = ax.get_legend_handles_labels()

    #fig.legend(handles, labels, loc='upper center', ncol=2)
    # Show the plot
    fig.tight_layout()
    fig.savefig('images/problem_statement.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    run()
