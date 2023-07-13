subdatasets=( loco-small-default loco-small-2 )
objects=( breakfast_box juice_bottle pushpins screw_bag splicing_connectors )

# Train
for subdataset in ${subdatasets[@]}
do
    for object in ${objects[@]}
    do
        python mvtec_loco_ad_evaluation/evaluate_experiment.py \
        --dataset_base_dir './mvtec_loco_anomaly_detection/' \
        --anomaly_maps_dir "./output/${subdataset}/anomaly_maps/mvtec_loco/" \
        --output_dir "./output/${subdataset}/metrics/mvtec_loco/" \
        --num_parallel_workers 8 \
        --object_name ${object} 
    done
done


