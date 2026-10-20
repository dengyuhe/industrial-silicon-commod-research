"""Shared paths and plotting style for the pipeline scripts."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# Sample used in the report: listing (22 Dec 2022) to 21 Jul 2025; train = first 498 trading days
END_DATE = "2025-07-21"
N_TRAIN = 498

INK, BLUE, ORANGE, GREY, RED = "#1b1f27", "#1f4fd8", "#e07b00", "#8a909c", "#c0392b"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.titlelocation": "left", "axes.titlesize": 10.5})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved figures/" + name)
