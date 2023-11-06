import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 1e9 0 62.2 min
# 0.15 817 0 max

df = pd.DataFrame({
    'alpha': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2],
    'refusal': [809, 777, 735, 680, 596, 503, 422, 334, 266, 216, 188, 149, 126, 109, 90, 78, 68, 59, 52, 47, 38],
    'accuracy': [87.5, 67.5, 67.1, 64.7, 63.6, 64.7, 66.0, 64.6, 65.1, 65.5, 65.4, 65.1, 64.8, 64.6, 64.2, 63.5, 63.4, 63.5, 63.4, 63.2, 63.1]
})

plt.rcParams["font.weight"] = "bold"

fig, ax1 = plt.subplots(figsize=(10, 6))

line1 = ax1.plot(df['alpha'], df['refusal'], color='dodgerblue', label='Refusal Number', linewidth=2)
ax2 = ax1.twinx()
line2 = ax2.plot(df['alpha'], df['accuracy'], color='salmon', label='Accuracy of MC1', linewidth=2)

ax1.set_title('Refusal Number and Accuracy of MC1 vs Alpha')
ax1.set_xlabel('Alpha')
ax1.set_ylabel('Refusal Number', color='dodgerblue')
ax2.set_ylabel('Accuracy of MC1', color='salmon')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

idx_075 = df[df['alpha'] == 0.75].index[0]
ax1.scatter(df['alpha'][idx_075], df['refusal'][idx_075], color='dodgerblue')
ax1.annotate('('+str(df['alpha'][idx_075])+', '+str(df['refusal'][idx_075])+')', (df['alpha'][idx_075], df['refusal'][idx_075]), textcoords="offset points", xytext=(0,25), ha='center', color='dodgerblue', weight='bold')
ax2.scatter(df['alpha'][idx_075], df['accuracy'][idx_075], color='salmon')
ax2.annotate('('+str(df['alpha'][idx_075])+', '+str(df['accuracy'][idx_075])+')', (df['alpha'][idx_075], df['accuracy'][idx_075]), textcoords="offset points", xytext=(0,-25), ha='center', color='salmon', weight='bold')

max_refusal_idx = df['refusal'].idxmax()
min_refusal_idx = df['refusal'].idxmin()
max_accuracy_idx = df['accuracy'].idxmax()
min_accuracy_idx = df['accuracy'].idxmin()

ax1.scatter(df['alpha'][max_refusal_idx], df['refusal'][max_refusal_idx], color='dodgerblue')
ax1.annotate('('+str(df['alpha'][max_refusal_idx])+', '+str(df['refusal'][max_refusal_idx])+')', (df['alpha'][max_refusal_idx], df['refusal'][max_refusal_idx]), textcoords="offset points", xytext=(10,5), ha='center', color='dodgerblue', weight='bold')
ax1.scatter(df['alpha'][min_refusal_idx], df['refusal'][min_refusal_idx], color='dodgerblue')
ax1.annotate('('+str(df['alpha'][min_refusal_idx])+', '+str(df['refusal'][min_refusal_idx])+')', (df['alpha'][min_refusal_idx], df['refusal'][min_refusal_idx]), textcoords="offset points", xytext=(-5,10), ha='center', color='dodgerblue', weight='bold')
ax2.scatter(df['alpha'][max_accuracy_idx], df['accuracy'][max_accuracy_idx], color='salmon')
ax2.annotate('('+str(df['alpha'][max_accuracy_idx])+', '+str(df['accuracy'][max_accuracy_idx])+')', (df['alpha'][max_accuracy_idx], df['accuracy'][max_accuracy_idx]), textcoords="offset points", xytext=(10,-20), ha='center', color='salmon', weight='bold')
ax2.scatter(df['alpha'][min_accuracy_idx], df['accuracy'][min_accuracy_idx], color='salmon')
ax2.annotate('('+str(df['alpha'][min_accuracy_idx])+', '+str(df['accuracy'][min_accuracy_idx])+')', (df['alpha'][min_accuracy_idx], df['accuracy'][min_accuracy_idx]), textcoords="offset points", xytext=(-10,-12), ha='center', color='salmon', weight='bold')

plt.show()