# subdatasets=( bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
subdatasets=( bottle cable capsule carpet grid )

# Train
for subdataset in ${subdatasets[@]}
do
    CUDA_VISIBLE_DEVICES=1 \
    python main-default.py \
    --yaml_config /workspace/Efficient-VQVAE/config/small-latent-8.yaml \
    --dataset mvtec_ad \
    --subdataset ${subdataset} \

done