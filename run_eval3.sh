subdatasets=( medium-default )

# Train
for subdataset in ${subdatasets[@]}
do
    python mvtec_ad_evaluation/evaluate_experiment.py \
    --dataset_base_dir './mvtec_anomaly_detection/' \
    --anomaly_maps_dir "./output/${subdataset}/anomaly_maps/mvtec_ad/" \
    --output_dir "./output/${subdataset}/metrics/mvtec_ad/" 
done


