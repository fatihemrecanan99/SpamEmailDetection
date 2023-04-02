# 2.1.1
print("--------------------------------------------------------------------")
print("Question 2.1.1")
import pandas as pd

y_train = pd.read_csv("y_train.csv")

numberofspam = y_train[y_train == 1].count()

allemails = y_train.count()

percentageofspam = (numberofspam / allemails) * 100

percentageofnonspam = 100 - percentageofspam.values[0]

threshold = 5

if abs(percentageofspam.values[0] - percentageofnonspam) <= threshold:
    print("Balanced.")
else:
    print("Skewed.")

print(f"Spam emails: {percentageofspam.values[0]:.2f}%")
print(f"Non-spam emails: {percentageofnonspam:.2f}%")

# Question 2.2
import pandas as pd
import numpy as np
print("--------------------------------------------------------------------")
print("Question 2.2")

x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

allsample = y_train.shape[0]
n_spam = sum(y_train.values)[0]
n_normal = allsample - n_spam
p_normal = n_normal / allsample
p_spam = n_spam / allsample

spamwords = np.sum(x_train[y_train.values.ravel() == 1], axis=0)
nonspmword = np.sum(x_train[y_train.values.ravel() == 0], axis=0)
allspamword = np.sum(spamwords)
allnormalword = np.sum(nonspmword)
thenormal = (nonspmword + 1) / (allnormalword + len(x_train.columns))
thespaml = (spamwords + 1) / (allspamword + len(x_train.columns))

normallogp = np.log(p_normal) + np.sum(np.log(thenormal) * x_test, axis=1)
spamlogp = np.log(p_spam) + np.sum(np.log(thespaml) * x_test, axis=1)
predictions = (spamlogp > normallogp).astype(int)

accuracy = np.mean(predictions == y_test.values.ravel())
confusion_matrix = pd.crosstab(y_test.values.ravel(), predictions, rownames=['Actual'], colnames=['Predicted'])

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(confusion_matrix)
wrong_predictions = sum(predictions != y_test.values.ravel())
print("Number of wrong predictions:", wrong_predictions)
print("------------------------------------------------------------------")
print("Question 2.3")

alpha = 5

thenormal_smoothed = (nonspmword + alpha) / (allnormalword + len(x_train.columns) * alpha)
thespaml_smoothed = (spamwords + alpha) / (allspamword + len(x_train.columns) * alpha)

normallogp_smoothed = np.log(p_normal) + np.sum(np.log(thenormal_smoothed) * x_test, axis=1)
spamlogp_smoothed = np.log(p_spam) + np.sum(np.log(thespaml_smoothed) * x_test, axis=1)
smoothedprediction = (spamlogp_smoothed > normallogp_smoothed).astype(int)

smoothedaccuray = np.mean(smoothedprediction == y_test.values.ravel())
smoothedconfmat = pd.crosstab(y_test.values.ravel(), smoothedprediction, rownames=['Actual'], colnames=['Predicted'])

print(f"Accuracy(additive smoothing):{smoothedaccuray:.4f}")
print("Confusion Matrix(additive smoothing):")
print(smoothedconfmat)

print("------------------------------------------------------------------")
print("Question 2.4")

x_train_binary = (x_train > 0).astype(int)
x_test_binary = (x_test > 0).astype(int)
N_normal = np.sum(y_train['Prediction'] == 0)
N_spam = np.sum(y_train['Prediction'] == 1)

prenormalword = x_train_binary[y_train['Prediction'] == 0].sum()
prespamword = x_train_binary[y_train['Prediction'] == 1].sum()

bernouillinormal = prenormalword / N_normal
bernouillispam = prespamword / N_spam

eps = 1e-12

normallogp_bernoulli = np.log(p_normal) + np.sum(np.log(np.maximum(bernouillinormal, eps)) * x_test_binary + np.log(np.maximum(1 - bernouillinormal, eps)) * (1 - x_test_binary), axis=1)
spamlogp_bernoulli = np.log(p_spam) + np.sum(np.log(np.maximum(bernouillispam, eps)) * x_test_binary + np.log(np.maximum(1 - bernouillispam, eps)) * (1 - x_test_binary), axis=1)
bernouillipredict = (spamlogp_bernoulli > normallogp_bernoulli).astype(int)


accuracy_bernoulli = np.mean(bernouillipredict == y_test.values.ravel())
confusion_matrix_bernoulli = pd.crosstab(y_test.values.ravel(), bernouillipredict, rownames=['Actual'], colnames=['Predicted'])


print(f"Accuracy(Bernoulli Naive Bayes): {accuracy_bernoulli: .4f}")
print("Confusion Matrix(Bernoulli Naive Bayes):")
print(confusion_matrix_bernoulli)

