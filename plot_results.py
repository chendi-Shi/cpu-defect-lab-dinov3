"""Generate a static experimental figure from measured validation records."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / 'reports' / 'VALIDATION_SWEEP.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    dimensions, banks = [32, 64, 128], [500, 2000, 5000]
    matrix = np.zeros((3,3))
    for row in rows:
        matrix[dimensions.index(int(row['dims'])), banks.index(int(row['bank_size']))] = float(row['validation_auroc'])
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    image = ax.imshow(matrix, cmap='YlGnBu', vmin=.8, vmax=1)
    ax.set_xticks(range(3), [str(x) for x in banks])
    ax.set_yticks(range(3), [str(x) for x in dimensions])
    ax.set_xlabel('Normal feature bank vectors')
    ax.set_ylabel('Feature dimensions')
    ax.set_title('Training-side synthetic validation AUROC', pad=16)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:.4f}', ha='center', va='center',
                    color='white' if matrix[i,j] > .93 else '#142334', fontsize=12)
    selected = json.loads((ROOT / 'reports' / 'selection.json').read_text(encoding='utf-8'))['selected']
    x, y = banks.index(selected['bank_size']), dimensions.index(selected['dims'])
    ax.add_patch(Rectangle((x-.5,y-.5), 1, 1, fill=False, edgecolor='#ff895a', linewidth=3))
    fig.colorbar(image, ax=ax, shrink=.85, label='Image AUROC')
    fig.tight_layout(rect=(0,.1,1,1))
    fig.text(.02, .025, '21 normal + 21 synthetic images. Orange outline: selected configuration; tie broken by bank bytes.',
             fontsize=8, color='#555555')
    fig.savefig(ROOT / 'reports' / 'VALIDATION_SWEEP.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
