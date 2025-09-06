# cbnu-acr-mlx-diff --git a/README.md b/README.md
index 9f9399a625b05d59b820226313c3e1d629978b4f..edab86c5a5b5a792f31e01167c30c4a8072141a6 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,59 @@
-# cbnu-acr-mlx-
+# CBNU Academic Rules LoRA fine-tuning (MLX)
+
+Fine-tune beomi/Llama-3-Open-Ko-8B with LoRA on the Chungbuk National University academic rules Q&A dataset using Apple's MLX framework.
+
+## Setup (macOS zsh)
+
+```zsh
+# create and activate virtual environment
+python3 -m venv .venv
+source .venv/bin/activate
+
+# install dependencies
+pip install -r requirements.txt
+
+# authenticate with Hugging Face
+export HUGGING_FACE_HUB_TOKEN=your_hf_token
+```
+
+> **Security**: store the token in environment variables or keychain, not in source files.
+
+## Training
+
+The script automatically creates `data/chungbuk_univ_academic_rules_QA.csv` with three dummy rows if it is missing.
+
+```zsh
+# optional: override base model
+export BASE_MODEL=beomi/Llama-3-Open-Ko-8B
+
+python train_acr.py \
+  --max_seq_length 512 \
+  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
+  --num_epochs 3 --batch_size 1 --learning_rate 2e-4
+```
+
+If memory is insufficient, try `--max_seq_length 384` or `256`, and `--lora_r 8`.
+
+## Inference
+
+```zsh
+# after training, run interactive chat applying the adapter
+python infer.py
+```
+
+Environment variables:
+- `BASE_MODEL` – default `beomi/Llama-3-Open-Ko-8B`
+- `ADAPTER_PATH` – LoRA adapter file (default `adapters.safetensors`)
+
+Both scripts use the fixed system prompt:
+> "너는 충북대학교 학칙 Q&A 어시스턴트다. 학칙에 근거해 정확·간결하게 한국어로 답한다."
+
+## Lint
+
+To check formatting locally:
+
+```zsh
+pip install black ruff
+black .
+ruff .
+```
