source ./training/paths.sh #run this script in the ai-persona directory

mlx_lm.generate \
  --model "$MODEL" \
  --adapter-path "$ADAPTER" \
  --system-prompt "$(cat "$SYSTEM_PROMPT")" \
  --prompt "Hi. How are you?"