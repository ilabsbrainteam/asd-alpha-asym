#!/bin/bash
set -euf -o pipefail

if [ $# -lt 2 ]
then
    echo "USAGE: run-pipeline.sh TASK STEPS [OTHER PASSTHROUGH ARGS]"
    echo "    allowed values for TASK:"
    echo "        caregiver, screen, staff"
    echo "    allowed values for STEPS:"
    echo "        pre, post (refers to wither ICA component selection is already done or not),"
    echo "        all (runs all init, preprocessing, and sensor steps),"
    echo "        or any valid MNE-BIDS-pipeline step name"
    exit 0
fi

case "$1" in
    caregiver | screen | staff)
        ;;
    *)
        echo "first argument must be one of: caregiver, screen, staff"
        exit 1
        ;;
esac

case "$2" in
    init | preprocessing | sensor)
        STEPS="$2"
        ;;
    pre)
        STEPS=init,preprocessing/_01_data_quality,preprocessing/_02_head_pos,preprocessing/_03_maxfilter,preprocessing/_04_frequency_filter,preprocessing/_05_regress_artifact,preprocessing/_06b_run_ssp,preprocessing/_07_make_epochs
        ;;
    post)
        STEPS=preprocessing/_08b_apply_ssp,preprocessing/_09_ptp_reject,sensor,source
        ;;
    all)
        STEPS=init,preprocessing,sensor
        ;;
    *)
        echo "second argument must be one of: pre, post, init, preprocessing, sensor, source, all"
        exit 1
        ;;
esac

mne_bids_pipeline --config "alpha_async_pipeline_config.py" --task=$1 --steps=$STEPS "${@:3}"
