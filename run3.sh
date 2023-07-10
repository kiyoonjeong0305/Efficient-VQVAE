subdatasets=( tile toothbrush transistor wood zipper)

# Train
for subdataset in ${subdatasets[@]}
do
    CUDA_VISIBLE_DEVICES=3 \
    python main.py \
    --yaml_config /workspace/Efficient-VQVAE/config/small-latent-8.yaml \
    --dataset mvtec_ad \
    --subdataset ${subdataset} \

done