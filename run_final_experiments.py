"""Run the locked validation-selected parameters, including a fresh category."""
import json
import subprocess
import sys
import time
from pathlib import Path
from lab import ROOT


def main():
    selected = json.loads((ROOT / 'reports' / 'selection.json').read_text(encoding='utf-8'))['selected']
    common = ['--dims', str(selected['dims']), '--bank-size', str(selected['bank_size'])]
    subprocess.run([sys.executable, str(ROOT / 'lab.py'), '--category', 'bottle', '--method', 'local',
                    '--threshold-split', 'reserved', *common], check=True, cwd=ROOT)
    deadline = time.monotonic() + 1800
    while not (ROOT / 'data' / 'mvtec' / 'provenance-screw.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Screw download did not finish. Re-run download_data.py --category screw')
        time.sleep(5)
    for method in ['global_shared', 'local']:
        subprocess.run([sys.executable, str(ROOT / 'lab.py'), '--category', 'screw', '--method', method,
                        *common], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / 'finalize.py')], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
