subdatasets=( bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)

# Train
for subdataset in ${subdatasets[@]}
do
    CUDA_VISIBLE_DEVICES=2 \
    python main.py \
    --yaml_config /workspace/Efficient-VQVAE/config/medium-latent-2-256.yaml \
    --dataset mvtec_ad \
    --subdataset ${subdataset} \

done