export HF_HOME=./.cache/huggingface/

baseline_dataset=gsm8k_cot_zeroshot
dataset=gsm8k_gen_star
model_id=models--mistralai--Mistral-7B-Instruct-v0.3
base_dir="/dccstor/cfm-tst/fms-sdg-internal"
iters=("0" "1" "2" "3")
base_model=$base_dir/.cache/huggingface/hub/$model_id/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db
model_prefix="$base_dir/output/gen_star_cot_transform/iter_" 
run_baseline=false

if [ "$run_baseline" = true ] ; then
    output_path=./results/$model_id/$baseline_dataset/baseline
    python -m lm_eval.__main__ --model vllm \
        --tasks $baseline_dataset \
        --model_args pretrained=$base_model \
        --batch_size auto \
        --log_samples \
        --output_path $output_path
fi

for iter in ${iters[@]}
do
    m_l=${model_prefix}${iter}/model
    mod_l=${m_l}_modified
    output_path=./results/$model_id/$dataset/iter_${iter}

    python ./scripts/reasoning/post_process_adapters_vLLM.py \
        --model_path $m_l \
        --output_model_path ${m_l}_modified

    for f in ${mod_l}/checkpoint-*
    do
        python -m lm_eval.__main__ --model vllm \
            --tasks $dataset \
            --model_args pretrained=$base_model,enable_lora=True,lora_local_path=$f \
            --batch_size auto \
            --log_samples \
            --output_path $output_path
    done
done
