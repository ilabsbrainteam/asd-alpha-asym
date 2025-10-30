from pathlib import Path

import matplotlib.pyplot as plt
import mne
import mne_bids
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

# path stuff
root = Path("/data/asd-alpha-asym")
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

# compute the spectra
spectra = dict()
for bp in bidspaths:
    raw = mne.io.read_raw_fif(bp.fpath, verbose=False)
    spec = raw.compute_psd(fmin=1, fmax=30, picks=roi_chs, verbose=False)
    spectra[(bp.subject, bp.task)] = spec
# aggregate in prep for stats
spectra_df = pd.DataFrame()
for (subj, task), spec in spectra.items():
    df = spec.to_data_frame(picks=list(set(roi_chs) - set(spec.info["bads"])))
    df[["subj", "task"]] = subj, task[4:]
    df["group"] = "ASD" if subj in asd_subjs else "TD"
    spectra_df = pd.concat((spectra_df, df))
# calculate left-minus-right asymmetry
roi_order = list()
for roi, (left, right) in rois.items():
    key = f"{roi}\n({left}-{right})"
    spectra_df[key] = spectra_df[left].sub(spectra_df[right])
    roi_order.append(key)
# drop individual electrode columns
spectra_df.drop(columns=roi_chs, inplace=True)
# restrict to alpha band & aggregate within-band power
alpha_power = spectra_df.loc[(8 <= spectra_df["freq"]) & (spectra_df["freq"] <= 12)]
alpha_power = (
    alpha_power.groupby(by=["subj", "task", "group"]).agg("mean").drop(columns="freq")
)
# in case we want only complete cases:
alpha_power_complete = alpha_power.dropna(axis=0, how="any")

# prepare for summary plot
var_name = "ROI"
value_name = "α-power asymmetry\n(left minus right; μV²)"
task_order = ("Caregiver", "Staff", "Screen")
group_order = ("TD", "ASD")

# convert to longform
alpha_power_long = pd.melt(
    alpha_power,
    ignore_index=False,
    var_name=var_name,
    value_name=value_name,
).reset_index()
alpha_power_long[value_name] *= 1e12  # convert V² to μV² for nicer plot
# force particular order on categorical vars
alpha_power_long["task"] = pd.Categorical(
    alpha_power_long["task"], categories=task_order, ordered=True
)
alpha_power_long["group"] = pd.Categorical(
    alpha_power_long["group"], categories=group_order, ordered=True
)
alpha_power_long.sort_values(
    by=["subj", "ROI", "task"], inplace=True, ignore_index=True
)
# complete cases: subj+ROI combo has non-NA data for all 3 tasks
complete = alpha_power_long.groupby(["subj", "ROI"]).filter(
    lambda g: (g[value_name].count() == len(task_order)) & ~g[value_name].isna().any()
)

for fname, _df in {"-complete-cases": complete, "": alpha_power_long}.items():
    # facet by ROI, complete cases vs all data
    p = (
        so.Plot(data=_df, x="task", y=value_name, color="subj", linestyle="group")
        .facet(col="ROI", order=roi_order)
        .add(so.Line(pointsize=6), so.Jitter(0.2), marker="subj")
        .theme({**sns.axes_style("whitegrid"), "grid.linestyle": ":"})
    )
    plotter = p.plot()
    fig = plotter._figure
    fig.set_size_inches(w=9, h=4)
    _ = [ax.set_xlabel("") for ax in fig.axes]
    fig.supxlabel("Task")
    # fig.legends[0].parent = fig.axes[-1]
    fig.legends[0].set_bbox_to_anchor((0.1, 0.6))
    fig.savefig(outdir / f"asymmetry-by-ROI{fname}.pdf")

    # Facet by ASD/TD
    p = (
        so.Plot(data=_df, x=var_name, y=value_name, color="task")
        .facet(col="group", order=group_order)
        .add(so.Dot(pointsize=6, alpha=0.5), so.Dodge(), so.Jitter(0.2), marker="subj")
        .add(so.Dash(), so.Agg(), so.Dodge())  # horz line: mean
        .add(
            so.Range(linewidth=2.5), so.Est(errorbar="se"), so.Dodge()
        )  # vert line: ±1 se
        .scale(color=so.Nominal("colorblind", order=task_order))
        .theme(
            {
                **sns.axes_style("whitegrid"),
                "grid.linestyle": ":",
                "legend.facecolor": "none",
            }
        )
    )
    plotter = p.plot()
    fig = plotter._figure
    fig.set_size_inches(w=10, h=4)
    _ = [ax.set_xlabel("") for ax in fig.axes]
    fig.supxlabel("ROI")
    fig.legends[0].parent = fig.axes[-1]
    fig.legends[0].set_bbox_to_anchor((0.65, 0.5 + (0.05 if fname else 0)))
    fig.savefig(outdir / f"asymmetry-by-group{fname}.pdf")

# subj×task spectrum plots
fig, axs = plt.subplots(
    len(tasks), len(subjs), sharex=True, sharey=True, layout="constrained"
)

ylim = (
    min([(10 * np.log10(s.data * 1e12)).min() for s in spectra.values()]),
    max([(10 * np.log10(s.data * 1e12)).max() for s in spectra.values()]),
)

for bp in bidspaths:
    # make the plot
    ax = axs[tasks.index(bp.task), subjs.index(bp.subject)]
    spec = spectra[(bp.subject, bp.task)]
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
        ax.set_ylim(ylim)
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
