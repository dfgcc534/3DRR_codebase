# 실험 노트북 셀 구조 표준

새 실험 노트북(`.ipynb`)을 만들 때 반드시 아래 순서를 따른다.

---

## 표준 셀 순서

| # | 종류 | 제목 | 내용 |
|---|------|------|------|
| 1 | code | Google Drive 마운트 | `drive.mount('/content/drive')` |
| 2 | code | 경로 설정 | `DRIVE_ROOT`, `CODEBASE_DIR`, `DATA_DIR`, `CONFIG_PATH` 정의 및 존재 확인 |
| 3 | code | 패키지 설치 | `%pip install -q gsplat torchvision omegaconf tqdm PyYAML` |
| 4 | code | Config 생성 | 실험별 YAML config를 딕셔너리로 작성 후 `CONFIG_PATH`에 저장 |
| 5 | code | 학습 실행 | `os.chdir(CODEBASE_DIR)` → `from train import train` → `train(CONFIG_PATH)` |
| 6 | code | 결과 시각화 | `outputs/.../examples/` 에서 `val_step*.jpg`, `train_aug.jpg` 인라인 표시 |
| 7 | code | 수치 평가 | `from core.evaluate import ...` → PSNR/SSIM/LPIPS 계산 → `metrics.json` 저장 |

---

## 규칙

- **Cell 2** 의 `DRIVE_ROOT` 만 수정하면 나머지 경로는 자동으로 맞춰진다.
- **Cell 4** 는 실험마다 달라지는 유일한 셀이다. 하이퍼파라미터/전처리 변경은 여기서만 한다.
- **Cell 7** 은 항상 `core/evaluate.py` 를 import해서 사용한다. 평가 로직을 노트북에 직접 쓰지 않는다.
- `outputs/` 는 `.gitignore` 에 포함되어 있으므로 커밋되지 않는다. 수치 결과는 `metrics.json` 으로 확인한다.

---

## Cell 7 표준 코드 (복붙용)

```python
%pip install -q lpips torchmetrics

import glob, os, sys

if CODEBASE_DIR not in sys.path:
    sys.path.insert(0, CODEBASE_DIR)

from core.evaluate import compute_metrics, save_metrics, print_metrics

outputs_root = os.path.join(CODEBASE_DIR, 'outputs')
test_dirs = sorted(glob.glob(os.path.join(outputs_root, '**', 'test'), recursive=True))
if not test_dirs:
    print('test 결과 폴더를 찾을 수 없습니다. 학습이 완료됐는지 확인하세요.')
else:
    pred_dir   = test_dirs[-1]
    output_dir = os.path.dirname(pred_dir)
    gt_dir     = os.path.join(DATA_DIR, 'test')

    print(f'pred_dir : {pred_dir}')
    print(f'gt_dir   : {gt_dir}')

    metrics = compute_metrics(pred_dir, gt_dir)
    save_metrics(metrics, output_dir)
    print_metrics(metrics)
```
