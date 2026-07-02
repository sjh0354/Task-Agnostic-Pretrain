model_name=nora
tasks=(
  bridge.sh
)

pip install -U transformers
pip install qwen_vl_utils

export CUDA_VISIBLE_DEVICES=1

ckpts=(
  /path/to/your/ckpt
)

action_ensemble_temp=-0.8
for ckpt_path in ${ckpts[@]}; do
  base_dir=$(dirname $ckpt_path)

  # evaluation in simulator
  # logging_dir=$base_dir/simpler_env/$(basename $ckpt_path)${action_ensemble_temp}
  logging_dir=results/$(basename $(dirname $ckpt_path))/$(basename $ckpt_path)${action_ensemble_temp}
  mkdir -p $logging_dir
  for i in ${!tasks[@]}; do
    task=${tasks[$i]}
    echo "🚀 running $task ..."
    device=1
    session_name=CUDA${device}-$(basename $logging_dir)-${task}
    bash scripts/$task $ckpt_path $model_name $action_ensemble_temp $logging_dir $device
  done

  # statistics evalution results
  echo "🚀 all tasks DONE! Calculating metrics..."
  python tools/calc_metrics_evaluation_videos.py \
    --log-dir-root $logging_dir \
    >>$logging_dir/total.metrics
done
