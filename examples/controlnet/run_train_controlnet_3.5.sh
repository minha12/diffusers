export MODEL_DIR="stabilityai/stable-diffusion-3.5-medium-diffusers"
export OUTPUT_DIR="sd3.5-controlnet-out"

accelerate launch train_controlnet_sd3.py \
    --pretrained_model_name_or_path=$MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --train_data_dir="fill50k" \
    --resolution=768 \
    --learning_rate=1e-5 \
    --dataset_preprocess_batch_size=1500 \
    --max_train_steps=15000 \
    --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --validation_steps=300 \
    --num_validation_images=4 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=4
