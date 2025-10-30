"""Create BIDS folder structure for "asd-alpha-async" data."""

import re
import yaml

from pathlib import Path
from warnings import filterwarnings

import mne

from mne_bids import BIDSPath, mark_channels, write_raw_bids

mne.set_log_level("WARNING")
# escalate SciPy warning so we can catch and handle it
filterwarnings(
    action="error",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning,
    module="scipy",
)
# escalate MNE-BIDS warning (we don't want to miss these)
filterwarnings(
    action="error",
    message="No events found or provided",
    category=RuntimeWarning,
    module="mne_bids",
)
# ignore missing coordinate info for misc channels
filterwarnings(
    action="ignore",
    message="No coordinate information found for channels",
    category=RuntimeWarning,
    module="mne"
)
filterwarnings(
    action="ignore",
    message="Not setting positions of",
    category=RuntimeWarning,
    module="mne"
)

# path stuff
root = Path("/data/asd-alpha-asym").resolve()
orig_data = root / "orig-data"
bids_root = root / "bids-data"
metadata = root / "metadata"
# outdir = root / "qc"

# # init logging (erase old log files)
# log = outdir / "bidsify-logfile.txt"
# with open(log, "w") as fid:
#     pass

# metadata
with open(metadata / "daysback.yaml") as fid:
    DAYSBACK = yaml.safe_load(fid)

with open(metadata / "bad-channels.yaml") as fid:
    prebads = yaml.safe_load(fid)

read_raw_kw = dict(preload=False, eog=("HEOG", "VEOG"), ignore_marker_types=True)
bids_path = BIDSPath(root=bids_root, datatype="eeg", suffix="eeg", extension=".eeg")

# classify raw files by "task" from the filenames
for data_folder in orig_data.iterdir():
    # extract the subject ID and group
    pattern = re.compile(r"OT(?P<subjnum>\d+)(?P<grp>-ASD)?", re.IGNORECASE)
    result = pattern.match(data_folder.name)
    asd = result.group("grp") is not None
    # BIDS requires subj to be a string, but cast to int as a failsafe first
    subj = f"{int(result.group("subjnum")):03}"
    bids_path.update(subject=subj)

    # classify the raw files by task, and write them to the BIDS folder
    for raw_file in data_folder.iterdir():
        if raw_file.suffix != ".vhdr":
            continue
        # extract task. files where subj looked at computer screen simply omit the
        # task from the filename
        pattern = r"(?:OXT_|005 )(?P<task>Caregiver|Staff)?"
        result = re.match(pattern, raw_file.name, re.IGNORECASE)
        task = (result.group("task") or "screen").lower()
        bids_path.update(task=task)
        # load the data, and write it in the BIDS folder tree
        raw = mne.io.read_raw_brainvision(raw_file, **read_raw_kw)
        ch_types = dict()
        if "HR" in raw.info["ch_names"]:
            ch_types.update(HR="ecg")
        if "GSR" in raw.info["ch_names"]:
            ch_types.update(GSR="bio")
        # set Fp1/2 as EOG channels if subjs don't have EOG
        if "eog" not in raw:
            ch_types.update(Fp1="eog", Fp2="eog")
        raw.set_channel_types(ch_types, on_unit_change="ignore")
        raw.pick(["eeg", "eog", "ecg", "bio"])  # TODO should we keep other channels?
        # drop events
        raw.set_annotations(None)
        bids_path.update(task=f"rest{task.capitalize()}")
        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            events=None,
            event_id=None,
            empty_room=None,
            anonymize=dict(daysback=DAYSBACK),
            overwrite=True,
        )
        # write the bad channels
        if subj in prebads and task in prebads[subj]:
            assert prebads[subj][task] is not None, (
                f"bad channels not yet marked in {raw_file}"
            )
            if prebads[subj][task]:
                mark_channels(
                    bids_path=bids_path,
                    ch_names=prebads[subj][task],
                    status="bad",
                    descriptions="prebad",
                )
        # print progress message to terminal
        print(f"{subj} {task: >10} completed")
