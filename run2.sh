subdatasets=( hazelnut leather metal_nut pill screw )

# Train
for subdataset in ${subdatasets[@]}
do
    CUDA_VISIBLE_DEVICES=2 \
    python main.py \
    --yaml_config /workspace/Efficient-VQVAE/config/small-latent-8.yaml \
    --dataset mvtec_ad \
    --subdataset ${subdataset} \

done