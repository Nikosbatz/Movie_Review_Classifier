import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve
from NB import NB
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report



def plot_learning_curve(estimator, title,
                        X_for_val, y_for_val,
                        X_for_test=None, y_for_test=None,
                        ylim=None,
                        val_cv=None,
                        test_cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    # print(train_sizes)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_for_val, y_for_val,
        cv=val_cv, n_jobs=-1, scoring='accuracy',
        train_sizes=train_sizes)

    # print(train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="b")
    plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.1,
                     color="green")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b",
             label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="green",
             label="Validation score")


    # _, _, test_scores = learning_curve(estimator,
    #                                    X_for_test, y_for_test,
    #                                    cv=test_cv, n_jobs=-1,
    #                                    scoring='accuracy',
    #                                    train_sizes=train_sizes)


    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    # plt.grid()


    # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                  test_scores_mean + test_scores_std, alpha=0.1,
    #                  color="red")

    # plt.plot(train_sizes, test_scores_mean, 'o-', color="red",
    #          label="Test score")

    plt.legend(loc="lower right")
    plt.show()
    return plt



"""nb = NB()
xTrain, yTrain = nb.loadTrainData()
xTrain, yTrain = nb.shuffleData(xTrain, yTrain)

nb = GaussianNB()

plot_learning_curve(estimator=nb, title='Learning Curve',
                    X_for_val=xTrain, 
                    y_for_val=yTrain, 
                    val_cv=None)"""


nb = NB()
xTrain, yTrain = nb.loadTrainData()
ls = nb.shuffleData(xTrain, yTrain)
xTrain = list(ls[0])
yTrain = list (ls[1])
nb = GaussianNB()
print(np.shape(xTrain))

"""nb.fit(xTrain, yTrain)
print(classification_report(yTrain, nb.predict(xTrain),
                            zero_division=1))"""





