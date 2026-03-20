import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

alphas = [0.6, 0.7, 0.8, 0.9]
aurocs = [0.56, 0.60, 0.57, 0.58]

plt.figure(figsize=(8,5))
plt.gca().set_facecolor('white')
plt.gcf().patch.set_facecolor('white')

plt.plot(alphas, aurocs, marker='o', markersize=8, linestyle='-', linewidth=2, color='#1e88e5')

for x, y in zip(alphas, aurocs):
    plt.annotate(f'{y:.2f}',
                 (x,y),
                 textcoords='offset points',
                 xytext=(0,10),
                 ha='center',
                 fontsize=16,
                 color='#333333')

plt.xlabel(r'alpha ($\alpha$)', fontsize=16)
plt.ylabel('AUROC', fontsize=16)

plt.xticks(alphas)
plt.ylim(0.54, 0.62)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

filename = "alpha_results_lineplot_20_steps.pdf"
plt.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')
print(f"Graph saved as: {filename}")

plt.show()
