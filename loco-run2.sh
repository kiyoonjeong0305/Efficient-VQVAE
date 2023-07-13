subdatasets=( breakfast_box juice_bottle pushpins screw_bag splicing_connectors )

# Train
for subdataset in ${subdatasets[@]}
do
    CUDA_VISIBLE_DEVICES=2 \
    python main-default.py \
    --yaml_config /workspace/Efficient-VQVAE/config/loco-medium-4.yaml \
    --dataset mvtec_loco \
    --subdataset ${subdataset} \

done