source paths.sh

mlx_lm.lora \
  --model "$MODEL" \
  --train \
  --data "$DATA" \
  --adapter-path "$ADAPTER" \
  --batch-size 4 \
  --iters 400 \
  --learning-rate 1e-5 \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --save-every 100 \
  --val-batches 25 \
  --mask-prompt