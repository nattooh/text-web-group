import matplotlib.pyplot as plt

models = ["DistilBERT", "BERT", "TF-IDF + LR", "BiLSTM"]
f1_scores = [0.9955, 0.9905, 0.9874, 0.9870]

plt.figure(figsize=(8,5))
bars = plt.bar(models, f1_scores)

plt.ylim(0.985, 1.0)  # 🔥 THIS is the key fix

plt.title("Model Performance (F1-macro)")
plt.ylabel("F1 Score")

# Add labels
for i, v in enumerate(f1_scores):
    plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')

plt.show()