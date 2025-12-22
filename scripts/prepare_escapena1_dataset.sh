#!/bin/bash

source .venv/bin/activate

source .env

python ${PROJECT_DIR}/mrs_prediction/dataset/prepare_dataset.py --dataset_name=escapena1 \
--dataset_dir=${DATASET_DIR} \
--modalities "B0_NCCT" "B0_CTA" \
--join=inner \
--id_column=usubjid \
--mrs_column="mrs_D90" \
--include_columns "siteid" "age" "mrs_BL" "baseline_nihss" "baseline_aspects" "onset_time_to_treatment" "male" "left_side_stroke" "ica_occlusion" "distal_m1_mca_occlusion" "proximal_m1_mca_occlusion" "mid_m1_mca_occlusion" "m2_m3_mca_occlusion" "occlusion_location" "history_of_hypertension" "history_of_smoking" "history_of_diabetes" "history_of_atrial_fibrilation" "history_of_peripheral_vascular_disease" "history_of_high_cholestrol" "history_of_congestive_heart_failure" "history_of_chronic_renal_failure" "history_of_major_surgery" "history_of_past_stroke" "history_of_recent_stroke" "history_of_coronary_artery_disease"  \
--test_split \
--skull_stripped