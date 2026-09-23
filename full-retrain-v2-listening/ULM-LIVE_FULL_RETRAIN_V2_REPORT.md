# ULM-LIVE FULL RETRAIN V2 REPORT

Git HEAD: aa6c543fbc06df8344e4470c590289d076de74e8
Runtime tree: pre-existing uncommitted evaluation cap of 250 frames (20 seconds) in `scripts/train_full_retrain_v2.py`; no code changes made during this task.

AWS: ap-northeast-2
Instance: i-0f732bf7d1cc409b4
GPU: NVIDIA L40S (46,068 MiB)

Training:
Epochs completed: 20
Optimizer steps: 27,700
Runtime: 1h 18m 57s summed epoch runtime (includes initial epochs and evaluations); 3,618s in the successful resumed process. One failed epoch-4 evaluation was retried with identical model weights.

Config:
Batch: 8
Grad accumulation: 2
Effective batch: 16
LR: 2e-4
Warmup: 0.03
Optimizer: AdamW
Scheduler: cosine
Dtype: bfloat16
Stop Pos Weight: 5.0
Stop Loss Weight: 0.05
Quantizers: 16

Epoch Table:

| Epoch | Train Codec | Val Codec | Q0 CE | Residual CE | Mean Token Accuracy | Q0 Accuracy | Residual Accuracy | Stop F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.0939 | 6.5620 | 5.5704 | 6.6319 | 3.916% | 8.325% | 3.622% | 39.255% |
| 2 | 6.4384 | 6.2977 | 4.7582 | 6.4047 | 5.179% | 13.566% | 4.620% | 39.569% |
| 3 | 6.2685 | 6.1791 | 4.4056 | 6.3017 | 5.933% | 16.544% | 5.225% | 38.937% |
| 4 | 6.1646 | 6.0924 | 4.2198 | 6.2216 | 6.455% | 17.907% | 5.692% | 38.332% |
| 5 | 6.0842 | 6.0361 | 4.1169 | 6.1685 | 6.704% | 18.685% | 5.905% | 38.610% |
| 6 | 6.0178 | 5.9962 | 4.0644 | 6.1293 | 6.964% | 19.293% | 6.142% | 38.434% |
| 7 | 5.9599 | 5.9685 | 4.0825 | 6.0985 | 7.090% | 19.243% | 6.280% | 39.245% |
| 8 | 5.9070 | 5.9523 | 4.1317 | 6.0781 | 7.184% | 19.130% | 6.388% | 39.151% |
| 9 | 5.8568 | 5.9517 | 4.2273 | 6.0709 | 7.222% | 18.649% | 6.460% | 37.851% |
| 10 | 5.8110 | 5.9558 | 4.3374 | 6.0679 | 7.233% | 18.352% | 6.492% | 37.333% |
| 11 | 5.7695 | 5.9676 | 4.4662 | 6.0719 | 7.226% | 17.962% | 6.511% | 37.749% |
| 12 | 5.7319 | 5.9775 | 4.5887 | 6.0743 | 7.207% | 17.533% | 6.519% | 37.555% |
| 13 | 5.6999 | 6.0004 | 4.7350 | 6.0889 | 7.176% | 17.190% | 6.509% | 37.098% |
| 14 | 5.6723 | 6.0111 | 4.8426 | 6.0932 | 7.166% | 16.954% | 6.513% | 37.286% |
| 15 | 5.6487 | 6.0289 | 4.9369 | 6.1059 | 7.126% | 16.650% | 6.491% | 37.217% |
| 16 | 5.6306 | 6.0430 | 5.0201 | 6.1154 | 7.098% | 16.465% | 6.473% | 37.417% |
| 17 | 5.6159 | 6.0527 | 5.0747 | 6.1221 | 7.097% | 16.456% | 6.473% | 36.944% |
| 18 | 5.6061 | 6.0599 | 5.1111 | 6.1273 | 7.085% | 16.372% | 6.466% | 37.231% |
| 19 | 5.6011 | 6.0628 | 5.1250 | 6.1295 | 7.075% | 16.311% | 6.460% | 37.253% |
| 20 | 5.5994 | 6.0638 | 5.1288 | 6.1304 | 7.076% | 16.309% | 6.460% | 37.226% |

Old Baseline:
Mean Token Accuracy: 5.874%
Q0 Accuracy: 15.932%
Codec CE: 6.0290

New Best:
Epoch: 10 (accuracy); epoch 9 (codec); epoch 6 (Q0 accuracy)
Mean Token Accuracy: 7.233%
Q0 Accuracy: 18.352% at epoch 10; 19.293% maximum at epoch 6
Codec CE: 5.9558 at epoch 10; 5.9517 minimum at epoch 9

Best Codec Checkpoint: /home/ubuntu/ULM-LIVE/outputs/talker-full-retrain-v2/best_codec.pt (epoch 9)
Best Accuracy Checkpoint: /home/ubuntu/ULM-LIVE/outputs/talker-full-retrain-v2/best_accuracy.pt (epoch 10)

Generation Evaluated Epochs: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
Evaluation limit: epoch 2 was generated before the pre-existing 250-frame cap; later evaluations used a 250-frame cap, so duration and stop comparisons with epoch 2 are not directly comparable.
Token Collapse: NO at epoch 20; repetition warnings at epochs 2, 4, 8, 14; all 20 sampling and all 10 greedy outputs unique at every evaluated epoch.
Stop Stability: epoch 20 stop F1 37.226%; 20/20 sampling and 10/10 greedy stopped by predicted stop. Test reconstruction mean generated/ground-truth length ratio 5.192, so timing remains weak.

Recommended Human Listening Candidates:
1. epoch 10 — best mean token accuracy, no repetition warning
2. epoch 8 — near-best codec loss, higher Q0 accuracy than epoch 10
3. epoch 6 — highest Q0 accuracy, no repetition warning

Human Listening Status: HUMAN LISTENING REQUIRED
ROOT RESULT: UNDERTRAINING RESOLVED relative to old objective baseline; later epochs overfit and recognizable Korean speech remains unverified.
PRODUCTION TALKER CANDIDATE: /home/ubuntu/ULM-LIVE/outputs/talker-full-retrain-v2/best_accuracy.pt
PRODUCTION READY = FALSE

Local Copy Validation: 10 evaluated epoch directories, 1,100 WAV files decoded, 1,549 files matched remote SHA-256 hashes.
AWS EC2 FINAL STATE: STOPPED
