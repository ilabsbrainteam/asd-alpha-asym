from pathlib import Path

import mne
import mne_bids

deriv_root = Path("bids-data") / "derivatives" / "mne-bids-pipeline"

bidspaths = mne_bids.find_matching_paths(
    deriv_root, processings="icafit", suffixes="epo", extensions=".fif"
)
