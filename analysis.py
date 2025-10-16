from pathlib import Path

import matplotlib.pyplot as plt
import mne
import mne_bids

deriv_root = Path("bids-data") / "derivatives" / "mne-bids-pipeline"

bidspaths = mne_bids.find_matching_paths(
    deriv_root,
    processings="clean",
    # suffixes="epo",
    suffixes="raw",
    extensions=".fif",
)

subjs = mne_bids.get_entity_vals(deriv_root, "subject")
tasks = mne_bids.get_entity_vals(deriv_root, "task")

fig, axs = plt.subplots(
    len(tasks), len(subjs), sharex=True, sharey=True, layout="constrained"
)

for bp in bidspaths:
    raw = mne.io.read_raw_fif(bp.fpath)
    spec = raw.compute_psd(fmin=1, fmax=20)
    ax = axs[tasks.index(bp.task), subjs.index(bp.subject)]
    spec.plot(axes=ax)

for rix, row in enumerate(axs):
    for cix, ax in enumerate(row):
        if rix == 0:
            ax.set_title(f"{subjs[cix]}\n" + ax.get_title())
        if cix == 0:
            ax.set_ylabel(f"{tasks[rix]}\n" + ax.get_ylabel())

fig.show()
