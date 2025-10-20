from pathlib import Path

import matplotlib.pyplot as plt
import mne
import mne_bids
import numpy as np
import pandas as pd
import seaborn as sns

# path stuff
root = Path(".").resolve()
deriv_root = root / "bids-data" / "derivatives" / "mne-bids-pipeline"
outdir = root / "qc"

# sensors of interest
# note that in montages based on the 10-10 system, T5 → P7 and T6 → P8
rois = dict(
    frontal=("F3", "F4"),
    central=("C3", "C4"),
    parietal_medial=("P3", "P4"),
    parietal_lateral=("P7", "P8"),
)
roi_chs = np.array(list(rois.values())).ravel().tolist()

# band cutoffs
band_cutoffs = dict(delta=(0, 4), theta=(4, 8), alpha=(8, 12), beta=(12, 30))

bidspaths = mne_bids.find_matching_paths(
    deriv_root,
    processings="clean",
    # suffixes="epo",
    suffixes="raw",
    extensions=".fif",
)

subjs = mne_bids.get_entity_vals(deriv_root, "subject")
tasks = mne_bids.get_entity_vals(deriv_root, "task")
asd_subjs = ("001", "002")

fig, axs = plt.subplots(
    len(tasks), len(subjs), sharex=True, sharey=True, layout="constrained"
)

spectra = dict()

for bp in bidspaths:
    raw = mne.io.read_raw_fif(bp.fpath)
    # plot from 0-30 Hz (delta, theta, alpha, beta)
    spec = raw.compute_psd(fmin=1, fmax=30, picks=roi_chs)
    spectra[(bp.subject, bp.task)] = spec
    # make the plot
    ax = axs[tasks.index(bp.task), subjs.index(bp.subject)]
    spec.plot(axes=ax)
    # shade the non-alpha bands
    for sides in (1, 8), (12, 30):
        ax.fill_between(
            sides, 0, 1, color="k", alpha=0.1, transform=ax.get_xaxis_transform()
        )
# fixup titles/ylabels, hide axes where data for that subj+task combination is missing
for rix, row in enumerate(axs):
    for cix, ax in enumerate(row):
        asd = " (ASD)" if subjs[cix] in asd_subjs else ""
        title = "" if rix > 0 else f"Subj-{subjs[cix]}{asd}\n{ax.get_title()}"
        ylabel = "" if cix > 0 else f"{tasks[rix][4:]}\n{ax.get_ylabel()}"
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if not len(ax.lines):
            ax.set_axis_off()
        else:
            if rix == len(axs) - 1:
                ticks = np.array([1, 4, 8, 12, 30])
                ax.set_xticks(ticks, labels=list(map(str, ticks)), fontsize="x-small")
                ax.set_xticks(
                    (ticks[1:] + ticks[:-1]) / 2,
                    labels=["δ", "θ", "α", "β"],
                    minor=True,
                )
                ax.xaxis.set_tick_params(which="major", length=12)
            ax.xaxis.set_tick_params(which="minor", length=0)


fig.set_size_inches(w=12, h=6)
fig.savefig(outdir / "spectra.pdf")
plt.close(fig)

# aggregate in prep for stats
spectra_df = pd.DataFrame()
for (subj, task), spec in spectra.items():
    df = spec.to_data_frame(picks=list(set(roi_chs) - set(spec.info["bads"])))
    df[["subj", "task", "asd"]] = subj, task[4:], subj in asd_subjs
    spectra_df = pd.concat((spectra_df, df))
# calculate left-minus-right asymmetry
for roi, chs in rois.items():
    spectra_df[f"{roi} ({chs[0]}-{chs[1]})"] = spectra_df[chs[0]].sub(
        spectra_df[chs[1]]
    )
# drop individual electrode columns
spectra_df.drop(columns=roi_chs, inplace=True)
# restrict to alpha band & aggregate within-band power
alpha_power = spectra_df.loc[(8 <= spectra_df["freq"]) & (spectra_df["freq"] <= 12)]
alpha_power = (
    alpha_power.groupby(by=["subj", "task", "asd"]).agg("mean").drop(columns="freq")
)

1 / 0
alpha_power_long = pd.melt(
    alpha_power,
    ignore_index=False,
    var_name="ROI (left minus right)",
    value_name="Power (μV²)",
).reset_index()

g = sns.catplot(
    data=alpha_power_long,
    x="ROI (left minus right)",
    y="Power (μV²)",
    hue="task",
    col="asd",
    dodge=True,
    kind="point",
    linestyle="",
    marker="_",
    markersize=10,
    order=alpha_power.columns,
)
g.map(sns.stripplot, dodge=True, size=20)  # TODO this doesn't work
g.figure.show()
