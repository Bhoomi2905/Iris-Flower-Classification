import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

df = pd.read_csv("iris.csv", header=None, names=columns)

print(df.head())

plt.scatter(df['petal_length'], df['petal_width'], c=pd.factorize(df['species'])[0])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Petal Length vs Petal Width")
plt.show()

df.hist(figsize=(10, 8))
plt.suptitle("Histogram of Iris Features")
plt.show()

df[df['species'] == 'Iris-setosa']['petal_length'].hist(alpha=0.5, label='Setosa')
df[df['species'] == 'Iris-versicolor']['petal_length'].hist(alpha=0.5, label='Versicolor')
df[df['species'] == 'Iris-virginica']['petal_length'].hist(alpha=0.5, label='Virginica')
plt.legend()
plt.title("Petal Length Distribution by Species")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

