#!/bin/bash
# Comparative benchmark: qwen3.5:4b vs qwen3.5:2b
# Tests: text reply, speed (tok/s), prefill time

set -e
cd "$(dirname "$0")/.."

PROMPTS=(
  "Hi, how are you today?"
  "Tell me something funny that happened to you, briefly."
  "What do you think of people who send many messages in a row without waiting for replies?"
)

run_test() {
  local model=$1
  local prompt=$2
  echo "--- model=$model prompt=\"${prompt:0:40}...\" ---"
  curl -s http://localhost:11434/api/generate -d "$(python3 -c "
import json
print(json.dumps({
  'model': '$model',
  'prompt': '''$prompt''',
  'stream': False,
  'think': False,
  'system': 'You are a friendly 24-year-old. Reply in English, short (1-2 sentences), informal and natural, like in a chat.',
  'options': {'num_predict': 80, 'num_ctx': 4096, 'temperature': 0.7}
}))")" | python3 -c "
import json, sys
d = json.load(sys.stdin)
resp = d.get('response','').strip()
toks = d.get('eval_count', 0)
dur = d.get('eval_duration', 1) / 1e9
prefill = d.get('prompt_eval_duration', 0) / 1e9
total = d.get('total_duration', 0) / 1e9
tps = toks / dur if dur > 0 else 0
print(f'  Resp: {resp}')
print(f'  Tokens gen: {toks} in {dur:.1f}s  => {tps:.2f} tok/s')
print(f'  Prefill: {prefill:.1f}s | Total: {total:.1f}s')
"
}

for MODEL in "qwen3.5:4b" "qwen3.5:2b"; do
  echo "================================"
  echo "TESTING $MODEL"
  echo "================================"
  # warmup (load model into RAM)
  curl -s http://localhost:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":1}}" > /dev/null
  for P in "${PROMPTS[@]}"; do
    run_test "$MODEL" "$P"
  done
  # unload model to free RAM before the next test
  curl -s http://localhost:11434/api/generate -d "{\"model\":\"$MODEL\",\"keep_alive\":0}" > /dev/null
  sleep 2
done
echo "================================"
echo "DONE"
