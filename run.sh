subdatasets=( bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper )
# subdatasets=( transistor wood zipper )

# Train
for subdataset in ${subdatasets[@]}
do

    python main.py \
    --yaml_config /workspace/EfficientAD-VQVAE/config/vqvae-dim-1.yaml \
    --dataset mvtec_ad \
    --subdataset ${subdataset} \

done