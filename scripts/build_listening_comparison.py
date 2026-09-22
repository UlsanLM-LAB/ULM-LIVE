#!/usr/bin/env python3
"""Build and validate a deterministic blind Talker listening comparison pack."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import random
import sys
from typing import Any
import wave

import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ulm_live.codec import build_codec
from ulm_live.talker import (
    SpeechSynthesizer,
    TalkerGenerationConfig,
    load_talker_checkpoint,
)
from ulm_live.thinker import ULMThinker


CHECKPOINTS = ("epoch-2", "epoch-3", "epoch-5")
LABELS = ("A", "B", "C")
SAMPLING_TEMPERATURE = 0.8
SAMPLING_TOP_P = 0.9
STOP_THRESHOLD = 0.50
BLIND_RANDOMIZATION_SEED = 20260922
SPEAKER = "DKSR20000953_1"
DIALECT = "ulsan"

# The ten fixed prompts from the full-training evaluation are retained where they
# fit the listening categories. The remaining prompts broaden dialect, length,
# and mixed-token coverage while keeping one fixed speaker and dialect.
PROMPTS: list[dict[str, Any]] = [
    {"prompt_id": "prompt_001", "category": "short", "text": "안녕", "source": "full_training_fixed"},
    {"prompt_id": "prompt_002", "category": "short", "text": "밥 묵었나?", "source": "full_training_fixed"},
    {"prompt_id": "prompt_003", "category": "short", "text": "오늘 날씨 좋네.", "source": "full_training_fixed"},
    {"prompt_id": "prompt_004", "category": "short", "text": "지금 어디고?", "source": "listening_pack"},
    {"prompt_id": "prompt_005", "category": "short", "text": "조심해서 가라.", "source": "listening_pack"},
    {"prompt_id": "prompt_006", "category": "conversational", "text": "오늘 뭐 하고 있었노?", "source": "full_training_fixed"},
    {"prompt_id": "prompt_007", "category": "conversational", "text": "나는 오늘 학교 끝나고 친구 만나러 간다.", "source": "full_training_fixed"},
    {"prompt_id": "prompt_008", "category": "conversational", "text": "오늘 기분이 좀 안 좋다.", "source": "full_training_fixed"},
    {"prompt_id": "prompt_009", "category": "conversational", "text": "주말에 시간 되면 같이 영화 보러 갈래?", "source": "listening_pack"},
    {"prompt_id": "prompt_010", "category": "conversational", "text": "점심은 먹었어? 아직이면 내가 맛있는 데 알려줄게.", "source": "listening_pack"},
    {"prompt_id": "prompt_011", "category": "ulsan_dialect", "text": "니 오늘 와 이리 늦었노?", "source": "listening_pack"},
    {"prompt_id": "prompt_012", "category": "ulsan_dialect", "text": "그거 내가 아까 말했제.", "source": "listening_pack"},
    {"prompt_id": "prompt_013", "category": "ulsan_dialect", "text": "날씨가 추우이 단디 입고 가라.", "source": "listening_pack"},
    {"prompt_id": "prompt_014", "category": "ulsan_dialect", "text": "마, 그 정도면 진짜 잘했다 아이가.", "source": "listening_pack"},
    {"prompt_id": "prompt_015", "category": "ulsan_dialect", "text": "울산에서 놀러 갈 만한 데 추천해줘.", "source": "full_training_fixed"},
    {"prompt_id": "prompt_016", "category": "ulsan_dialect", "text": "태화강 쪽으로 산책하러 갈라 카는데 같이 갈래?", "source": "listening_pack"},
    {"prompt_id": "prompt_017", "category": "ulsan_dialect", "text": "아무리 바빠도 밥은 챙겨 묵어야 된다.", "source": "listening_pack"},
    {"prompt_id": "prompt_018", "category": "ulsan_dialect", "text": "와 그라노, 무슨 일 있는 거 아이제?", "source": "listening_pack"},
    {"prompt_id": "prompt_019", "category": "ulsan_dialect", "text": "오늘 장생포 가면 바람이 많이 불라나?", "source": "listening_pack"},
    {"prompt_id": "prompt_020", "category": "ulsan_dialect", "text": "니가 괜찮다 카면 내도 마음이 놓인다.", "source": "listening_pack"},
    {"prompt_id": "prompt_021", "category": "long_explanatory", "text": "하늘이 파란 이유를 간단하게 설명해줘.", "source": "full_training_fixed"},
    {"prompt_id": "prompt_022", "category": "long_explanatory", "text": "울산의 태화강 국가정원이 시민들에게 어떤 의미가 있는지 처음 방문한 사람에게 차근차근 설명해줘.", "source": "listening_pack"},
    {"prompt_id": "prompt_023", "category": "long_explanatory", "text": "비가 많이 오는 날에는 길이 미끄럽고 시야도 좁아지니까 평소보다 천천히 운전하고 안전거리를 충분히 확보해야 한다.", "source": "listening_pack"},
    {"prompt_id": "prompt_024", "category": "long_explanatory", "text": "친구와 의견이 다를 때는 먼저 상대방의 말을 끝까지 듣고, 내가 이해한 내용을 확인한 다음 차분하게 내 생각을 말하는 것이 좋다.", "source": "listening_pack"},
    {"prompt_id": "prompt_025", "category": "long_explanatory", "text": "아침에 일어나서 물을 한 잔 마시고 가볍게 몸을 풀면 잠든 동안 굳어 있던 몸이 깨어나 하루를 조금 더 편안하게 시작할 수 있다.", "source": "listening_pack"},
    {"prompt_id": "prompt_026", "category": "numbers_english_technical", "text": "1 더하기 1은 2다.", "source": "full_training_fixed"},
    {"prompt_id": "prompt_027", "category": "numbers_english_technical", "text": "Python에서 리스트를 정렬하는 방법 알려줘.", "source": "full_training_fixed"},
    {"prompt_id": "prompt_028", "category": "numbers_english_technical", "text": "오늘 회의는 오후 3시 30분에 시작하고 Zoom 링크는 이메일로 보낼게.", "source": "listening_pack"},
    {"prompt_id": "prompt_029", "category": "numbers_english_technical", "text": "GPU 사용률이 95퍼센트이고 메모리는 24GB 중 18GB를 사용하고 있다.", "source": "listening_pack"},
    {"prompt_id": "prompt_030", "category": "numbers_english_technical", "text": "API 응답 시간이 120밀리초를 넘으면 timeout 로그를 확인해줘.", "source": "listening_pack"},
]

SCORE_FIELDS = ("clarity", "naturalness", "dialect", "voice_stability", "overall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thinker", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_stats(tokens: torch.Tensor) -> tuple[float, float]:
    q0 = tokens.squeeze(0)[0] if tokens.ndim == 3 else tokens[0]
    values = q0.detach().cpu().tolist()
    counts = Counter(values)
    total = len(values)
    if not total:
        return 0.0, 0.0
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    top1 = counts.most_common(1)[0][1] / total
    return round(entropy, 3), round(top1, 4)


def build_blind_key() -> dict[str, Any]:
    rng = random.Random(BLIND_RANDOMIZATION_SEED)
    mappings: dict[str, dict[str, str]] = {}
    for prompt in PROMPTS:
        shuffled = list(CHECKPOINTS)
        rng.shuffle(shuffled)
        mappings[prompt["prompt_id"]] = dict(zip(LABELS, shuffled, strict=True))
    return {"blind_randomization_seed": BLIND_RANDOMIZATION_SEED, "mappings": mappings}


def validate_wav(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 44:
        raise ValueError(f"missing or empty WAV: {path}")
    with path.open("rb") as stream:
        if stream.read(4) != b"RIFF" or stream.read(4) == b"":
            raise ValueError(f"invalid RIFF header: {path}")
        stream.seek(8)
        if stream.read(4) != b"WAVE":
            raise ValueError(f"invalid WAVE header: {path}")
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        sample_width = wav.getsampwidth()
        raw = wav.readframes(frames)
    if channels != 1:
        raise ValueError(f"expected mono WAV, got {channels} channels: {path}")
    if sample_rate != 24000:
        raise ValueError(f"expected 24000 Hz, got {sample_rate}: {path}")
    if frames <= 0 or not raw:
        raise ValueError(f"zero-duration WAV: {path}")
    dtype_by_width = {1: np.uint8, 2: "<i2", 4: "<i4"}
    if sample_width not in dtype_by_width:
        raise ValueError(f"unsupported PCM width {sample_width}: {path}")
    samples = np.frombuffer(raw, dtype=dtype_by_width[sample_width])
    if samples.size == 0 or not np.isfinite(samples.astype(np.float64)).all():
        raise ValueError(f"non-finite or empty samples: {path}")
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_frames": frames,
        "duration": frames / sample_rate,
        "nan_count": 0,
        "inf_count": 0,
    }


def metadata_matches(item: dict[str, Any], expected: dict[str, Any]) -> bool:
    keys = (
        "prompt_id", "checkpoint", "seed", "speaker", "text", "mode",
        "temperature", "top_p", "stop_threshold", "wav_path",
    )
    return all(item.get(key) == expected.get(key) for key in keys)


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def public_prompts() -> dict[str, Any]:
    return {
        "sampling": [
            {**prompt, "seed": 1000 + index, "speaker": SPEAKER, "dialect": DIALECT}
            for index, prompt in enumerate(PROMPTS, 1)
        ],
        "greedy": [
            {**prompt, "seed": None, "speaker": SPEAKER, "dialect": DIALECT}
            for prompt in PROMPTS[:10]
        ],
    }


def score_header() -> list[str]:
    columns = ["prompt_id", "text"]
    for label in LABELS:
        columns.extend(f"{label}_{field}" for field in SCORE_FIELDS)
    return columns + ["notes"]


def write_score_sheet(path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=score_header())
        writer.writeheader()
        for mode, prompts in public_prompts().items():
            for prompt in prompts:
                writer.writerow({"prompt_id": f"{mode}_{prompt['prompt_id']}", "text": prompt["text"]})


def write_readme(path: Path) -> None:
    path.write_text(
        """# ULM-LIVE Blind Listening Comparison Pack

이 비교팩은 epoch-2, epoch-3, epoch-5 Talker checkpoint를 블라인드로 비교합니다. `index.html`을 브라우저에서 직접 열어 Sampling 또는 Greedy를 선택하고 A/B/C를 평가하세요. 별도 서버나 설치가 필요하지 않습니다.

## 점수 기준

- 1 = 매우 나쁨
- 2 = 나쁨
- 3 = 보통
- 4 = 좋음
- 5 = 매우 좋음

평가 항목:

- Clarity: 발음이 잘 들리고 원문을 알아들을 수 있는가
- Naturalness: 로봇음, 이상한 반복, 부자연스러운 리듬이 적은가
- Dialect / Prosody: 울산 말투와 억양이 자연스러운가
- Voice Stability: 발화 중 음색/볼륨/피치가 갑자기 무너지지 않는가
- Overall: 전체적으로 다시 선택하고 싶은 음성인가

Sampling은 prompt별 seed가 고정되어 세 checkpoint에 똑같이 적용됩니다. Greedy는 sampling을 사용하지 않습니다. 평가 중에는 `blind_key.json`과 `generation_metadata.json`을 열지 마세요. 두 파일은 평가 완료 후 결과 해석과 재현성 검증에만 사용합니다.

웹 UI의 입력은 브라우저 localStorage에 자동 저장됩니다. 완료 후 CSV와 JSON을 모두 export해 보관하세요. 기본 `listening_scores.csv`는 수기 입력용 빈 양식입니다.
""",
        encoding="utf-8",
    )


def write_index(path: Path) -> None:
    prompts_json = json.dumps(public_prompts(), ensure_ascii=False).replace("</", "<\\/")
    fields_json = json.dumps(SCORE_FIELDS)
    document = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ULM-LIVE Blind Listening</title>
<style>
:root { color-scheme: light; --ink:#18202b; --muted:#657083; --line:#d8dee8; --accent:#165dff; --bg:#f4f6f9; }
* { box-sizing:border-box; } body { margin:0; font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif; color:var(--ink); background:var(--bg); }
main { width:min(1100px, calc(100% - 32px)); margin:32px auto; }
header,.panel,.candidate { background:white; border:1px solid var(--line); border-radius:12px; }
header { padding:20px 24px; display:flex; gap:16px; justify-content:space-between; align-items:center; flex-wrap:wrap; }
h1 { font-size:1.35rem; margin:0 0 5px; } .muted { color:var(--muted); font-size:.9rem; }
button,select,textarea { font:inherit; } button { border:1px solid var(--line); background:white; border-radius:8px; padding:9px 13px; cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:white; } button:disabled { opacity:.45; cursor:not-allowed; }
.toolbar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.panel { margin-top:16px; padding:22px; } .prompt-id { color:var(--accent); font-weight:700; } .prompt { font-size:1.2rem; line-height:1.65; margin:10px 0 4px; }
.category { color:var(--muted); font-size:.85rem; }
.grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:16px; }
.candidate { padding:16px; } .candidate h2 { margin:0 0 12px; font-size:1.25rem; } audio { width:100%; }
.scores { margin-top:14px; display:grid; grid-template-columns:1fr auto; gap:8px 10px; align-items:center; }
.scores label { font-size:.88rem; } .scores select { min-width:72px; padding:6px; border:1px solid var(--line); border-radius:6px; }
textarea { width:100%; min-height:84px; padding:10px; border:1px solid var(--line); border-radius:8px; resize:vertical; }
.notes { margin-top:18px; } .nav { display:flex; justify-content:space-between; align-items:center; gap:10px; margin-top:18px; }
.progress { min-width:180px; text-align:center; } progress { width:100%; }
@media(max-width:800px){ .grid { grid-template-columns:1fr; } main { width:min(100% - 20px,1100px); margin:10px auto; } }
</style>
</head>
<body><main>
<header><div><h1>ULM-LIVE Blind Listening</h1><div class="muted">A/B/C를 독립적으로 듣고 각 항목을 1–5점으로 평가하세요.</div></div>
<div class="toolbar"><label for="mode">모드</label><select id="mode"><option value="sampling">Sampling</option><option value="greedy">Greedy</option></select><button id="csv">CSV export</button><button id="json">JSON export</button></div></header>
<section class="panel"><div class="prompt-id" id="promptId"></div><div class="prompt" id="promptText"></div><div class="category" id="category"></div>
<div class="grid" id="candidates"></div>
<div class="notes"><label for="notes">Notes</label><textarea id="notes" placeholder="반복, 발음 오류, 억양, 선호 이유 등을 기록하세요."></textarea></div>
<div class="nav"><button id="prev">이전</button><div class="progress"><div id="progressText"></div><progress id="progress" max="1"></progress></div><button class="primary" id="next">다음</button></div></section>
</main>
<script>
const PROMPTS=__PROMPTS_JSON__;
const FIELDS=__FIELDS_JSON__;
const LABELS=['A','B','C'];
const FIELD_NAMES={clarity:'Clarity',naturalness:'Naturalness',dialect:'Dialect / Prosody',voice_stability:'Voice Stability',overall:'Overall'};
const key='ulm-live-listening-v1'; let mode='sampling', index=0;
let state={}; try { state=JSON.parse(localStorage.getItem(key)||'{}'); } catch (_) { state={}; }
const $=id=>document.getElementById(id);
function recordKey(m,p){ return `${m}:${p.prompt_id}`; }
function currentRecord(){ const p=PROMPTS[mode][index], k=recordKey(mode,p); if(!state[k]) state[k]={scores:{},notes:''}; return state[k]; }
function save(){ localStorage.setItem(key,JSON.stringify(state)); }
function complete(rec){ return LABELS.every(l=>FIELDS.every(f=>rec.scores?.[l]?.[f])); }
function render(){ const items=PROMPTS[mode], p=items[index], rec=currentRecord(); $('promptId').textContent=`${mode.toUpperCase()} · ${p.prompt_id}`; $('promptText').textContent=p.text; $('category').textContent=p.category;
  $('candidates').replaceChildren(...LABELS.map(label=>{ const box=document.createElement('article'); box.className='candidate'; const title=document.createElement('h2'); title.textContent=label; const audio=document.createElement('audio'); audio.controls=true; audio.preload='metadata'; audio.src=`${mode}/${p.prompt_id}/${label}.wav`; const scores=document.createElement('div'); scores.className='scores';
    FIELDS.forEach(field=>{ const lab=document.createElement('label'); lab.textContent=FIELD_NAMES[field]; lab.htmlFor=`${label}-${field}`; const sel=document.createElement('select'); sel.id=lab.htmlFor; sel.innerHTML='<option value="">-</option>'+[1,2,3,4,5].map(v=>`<option>${v}</option>`).join(''); sel.value=rec.scores?.[label]?.[field]||''; sel.onchange=()=>{ rec.scores[label]??={}; rec.scores[label][field]=sel.value; save(); updateProgress(); }; scores.append(lab,sel); }); box.append(title,audio,scores); return box; }));
  $('notes').value=rec.notes||''; $('prev').disabled=index===0; $('next').disabled=index===items.length-1; updateProgress(); }
function updateProgress(){ const items=PROMPTS[mode], done=items.filter(p=>complete(state[recordKey(mode,p)]||{})).length; $('progressText').textContent=`${index+1} / ${items.length} · 완료 ${done}`; $('progress').max=items.length; $('progress').value=done; }
$('notes').oninput=e=>{ currentRecord().notes=e.target.value; save(); }; $('prev').onclick=()=>{ if(index>0){index--;render();} }; $('next').onclick=()=>{if(index<PROMPTS[mode].length-1){index++;render();}};
$('mode').onchange=e=>{mode=e.target.value;index=0;render();};
function rows(){ return Object.entries(PROMPTS).flatMap(([m,items])=>items.map(p=>{const r=state[recordKey(m,p)]||{scores:{},notes:''}; return {mode:m,prompt_id:p.prompt_id,text:p.text,...Object.fromEntries(LABELS.flatMap(l=>FIELDS.map(f=>[`${l}_${f}`,r.scores?.[l]?.[f]||'']))),notes:r.notes||''};})); }
function download(name,type,data){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([data],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
$('csv').onclick=()=>{const rs=rows(), heads=['mode','prompt_id','text',...LABELS.flatMap(l=>FIELDS.map(f=>`${l}_${f}`)),'notes'];const esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';download('listening_scores_completed.csv','text/csv;charset=utf-8','\ufeff'+[heads,...rs.map(r=>heads.map(h=>r[h]))].map(row=>row.map(esc).join(',')).join('\n'));};
$('json').onclick=()=>download('listening_scores_completed.json','application/json',JSON.stringify({exported_at:new Date().toISOString(),results:rows()},null,2));
render();
</script></body></html>'''
    document = document.replace("__PROMPTS_JSON__", prompts_json).replace("__FIELDS_JSON__", fields_json)
    path.write_text(document, encoding="utf-8")


def expected_metadata(
    prompt: dict[str, Any], mode: str, checkpoint: str, label: str
) -> dict[str, Any]:
    prompt_number = int(prompt["prompt_id"].split("_")[-1])
    return {
        "prompt_id": prompt["prompt_id"],
        "checkpoint": checkpoint,
        "seed": 1000 + prompt_number if mode == "sampling" else None,
        "speaker": SPEAKER,
        "text": prompt["text"],
        "mode": mode,
        "temperature": SAMPLING_TEMPERATURE if mode == "sampling" else None,
        "top_p": SAMPLING_TOP_P if mode == "sampling" else None,
        "stop_threshold": STOP_THRESHOLD,
        "wav_path": f"{mode}/{prompt['prompt_id']}/{label}.wav",
    }


def validate_fairness(metadata: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in metadata:
        grouped.setdefault((item["mode"], item["prompt_id"]), []).append(item)
    expected_groups = 30 + 10
    if len(grouped) != expected_groups:
        raise ValueError(f"expected {expected_groups} comparison groups, got {len(grouped)}")
    compare_keys = ("text", "speaker", "seed", "temperature", "top_p", "stop_threshold")
    for key, items in grouped.items():
        if len(items) != 3 or {item["checkpoint"] for item in items} != set(CHECKPOINTS):
            raise ValueError(f"incomplete candidates for {key}")
        for field in compare_keys:
            if len({json.dumps(item[field], sort_keys=True) for item in items}) != 1:
                raise ValueError(f"fairness mismatch for {key}: {field}")
    sampling = [item for item in metadata if item["mode"] == "sampling"]
    if len(sampling) != 90 or any(item["seed"] is None for item in sampling):
        raise ValueError("sampling seed control is incomplete")


def validate_pack(output_dir: Path, metadata: list[dict[str, Any]]) -> dict[str, int]:
    validate_fairness(metadata)
    counts = {"sampling": 0, "greedy": 0}
    expected_paths = set()
    for item in metadata:
        path = output_dir / item["wav_path"]
        validate_wav(path)
        expected_paths.add(path.resolve())
        counts[item["mode"]] += 1
    actual_paths = {path.resolve() for mode in counts for path in (output_dir / mode).glob("prompt_*/*.wav")}
    if actual_paths != expected_paths:
        raise ValueError("WAV tree contains missing or unexpected files")
    if counts != {"sampling": 90, "greedy": 30}:
        raise ValueError(f"incorrect WAV counts: {counts}")
    required = ("prompts.json", "blind_key.json", "listening_scores.csv", "generation_metadata.json", "README.md", "index.html")
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing pack files: {missing}")
    return counts


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "generation_metadata.json"
    existing: list[dict[str, Any]] = []
    if metadata_path.is_file():
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing = value.get("samples", value if isinstance(value, list) else [])
    metadata_by_key = {
        (item.get("mode"), item.get("prompt_id"), item.get("checkpoint")): item
        for item in existing
    }

    blind_key = build_blind_key()
    write_json(output_dir / "blind_key.json", blind_key)
    write_json(output_dir / "prompts.json", public_prompts())
    write_score_sheet(output_dir / "listening_scores.csv")
    write_readme(output_dir / "README.md")
    write_index(output_dir / "index.html")
    print("blind_key.json saved successfully")

    if not args.validate_only:
        checkpoint_paths = {name: args.checkpoint_dir / f"{name}.pt" for name in CHECKPOINTS}
        missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing checkpoints: {missing}")

        print("Loading Thinker and Mimi codec...")
        thinker = ULMThinker(args.thinker, device=args.device, torch_dtype="bf16", freeze=True)
        codec = build_codec(backend="mimi", device=args.device)

        for checkpoint, checkpoint_path in checkpoint_paths.items():
            print(f"Generating candidate checkpoint {checkpoint}...")
            talker, checkpoint_metadata = load_talker_checkpoint(checkpoint_path, device=args.device)
            speaker2id = checkpoint_metadata.get("speaker2id", {})
            dialect2id = checkpoint_metadata.get("dialect2id", {})
            if SPEAKER not in speaker2id or DIALECT not in dialect2id:
                raise ValueError(f"required speaker/dialect absent from {checkpoint}")
            synthesizer = SpeechSynthesizer(
                thinker=thinker,
                talker=talker,
                codec=codec,
                speaker2id=speaker2id,
                dialect2id=dialect2id,
                device=args.device,
            )
            jobs = (("sampling", PROMPTS), ("greedy", PROMPTS[:10]))
            for mode, prompts in jobs:
                config = TalkerGenerationConfig(
                    max_new_tokens=750,
                    do_sample=(mode == "sampling"),
                    temperature=SAMPLING_TEMPERATURE if mode == "sampling" else 1.0,
                    top_p=SAMPLING_TOP_P if mode == "sampling" else None,
                    stop_threshold=STOP_THRESHOLD,
                    return_details=True,
                )
                for prompt in prompts:
                    label = next(
                        candidate_label
                        for candidate_label, candidate_checkpoint in blind_key["mappings"][prompt["prompt_id"]].items()
                        if candidate_checkpoint == checkpoint
                    )
                    expected = expected_metadata(prompt, mode, checkpoint, label)
                    wav_path = output_dir / expected["wav_path"]
                    prior = metadata_by_key.get((mode, prompt["prompt_id"], checkpoint))
                    if prior and metadata_matches(prior, expected):
                        try:
                            validate_wav(wav_path)
                            continue
                        except (ValueError, wave.Error):
                            pass
                    wav_path.parent.mkdir(parents=True, exist_ok=True)
                    if expected["seed"] is not None:
                        seed_everything(expected["seed"])
                    result = synthesizer.synthesize(
                        prompt["text"],
                        speaker=SPEAKER,
                        dialect=DIALECT,
                        generation_config=config,
                    )
                    result.save(wav_path)
                    wav_info = validate_wav(wav_path)
                    q0_entropy, q0_top1_ratio = token_stats(result.codec_tokens)
                    item = {
                        **expected,
                        "duration": round(wav_info["duration"], 3),
                        "frames": int(result.num_audio_frames),
                        "termination_reason": result.termination_reason,
                        "max_stop_probability": round(result.max_stop_prob, 4),
                        "Q0_entropy": q0_entropy,
                        "Q0_top1_ratio": q0_top1_ratio,
                    }
                    metadata_by_key[(mode, prompt["prompt_id"], checkpoint)] = item
                    ordered = sorted(metadata_by_key.values(), key=lambda x: (x["mode"], x["prompt_id"], x["checkpoint"]))
                    write_json(metadata_path, {"samples": ordered})
            del synthesizer, talker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metadata = sorted(metadata_by_key.values(), key=lambda x: (x["mode"], x["prompt_id"], x["checkpoint"]))
    write_json(metadata_path, {"samples": metadata})
    counts = validate_pack(output_dir, metadata)
    print(f"Sampling WAV count: {counts['sampling']}")
    print(f"Greedy WAV count: {counts['greedy']}")
    print("FAIR COMPARISON CHECK: PASS")
    print("SAMPLING SEED CONTROL: PASS")
    print("AUDIO INTEGRITY: PASS")


if __name__ == "__main__":
    main()
