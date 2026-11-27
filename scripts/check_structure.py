"""Kiểm tra cấu trúc project và phiên bản code so với tài liệu trong doc/.

Chạy từ thư mục energy_ts:   python scripts/check_structure.py
Thoát với mã 1 nếu có lỗi (thiếu file bắt buộc, code là bản cũ, doc thiếu mục, doc trỏ tới file không tồn tại).
Phần "kết quả đã sinh" chỉ thông tin, không tính là lỗi.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md", "requirements.txt", "configs/config.yaml", "data/raw/README.md",
    "src/__init__.py", "src/config.py", "src/utils.py", "src/metrics.py", "src/data.py", "src/eda.py",
    "src/preprocess.py", "src/baselines.py", "src/models.py", "src/train.py", "src/monitor.py",
    "src/evaluate.py", "src/attention.py", "src/training_report.py", "src/compare_ablation.py",
    "scripts/run_all.sh", "scripts/run_all.ps1", "scripts/make_synthetic.py",
    "scripts/run_ablation_no_temp.ps1", "scripts/run_ablation_no_temp.sh", "scripts/check_structure.py",
    "tests/test_data.py", "tests/test_models.py", "notebooks/report.ipynb",
] + [f"doc/{n}.md" for n in (
    "00_index", "01_overview", "02_io_conventions", "03_data_setup", "04_eda", "05_preprocessing", "06_baselines",
    "07_models", "08_training", "09_evaluation", "10_attention_analysis", "11_decisions_log", "12_runbook",
    "13_notebook_report", "14_ablation_no_temperature")]

# (file, chuỗi phải có, ý nghĩa nếu thiếu)
VERSION_MARKERS = [
    ("src/data.py", "exclude_features", "data.py là bản cũ: chưa hỗ trợ ablation (--set data.exclude_features sẽ bị BỎ QUA im lặng)"),
    ("src/train.py", "elapsed_s", "train.py là bản cũ: history.csv không ghi mỗi epoch, monitor không hoạt động"),
    ("src/evaluate.py", "BẮT BUỘC", "evaluate.py là bản cũ: chưa có kiểm tra Ridge bắt buộc (V3)"),
    ("configs/config.yaml", "exclude_features", "config.yaml chưa có data.exclude_features (không bắt buộc nhưng nên có)"),
]

GENERATED = [
    "data/processed/split_info.json", "data/processed/scaler.json", "outputs/eda/eda_summary.md",
    "outputs/baselines/metrics.json", "outputs/models/train_summary_all.csv", "outputs/metrics/summary.csv",
    "outputs/results_report.md", "outputs/attention/attention_summary.json",
    "outputs/figures/training_overview.png", "outputs/models/training_runs_table.csv",
    "data/processed_no_temp/feature_names.json", "outputs_no_temp/ablation_comparison.md",
]

PATH_RE = re.compile(r"`((?:src|doc|scripts|configs|tests|notebooks)/[A-Za-z0-9_./\-]+\.[A-Za-z0-9]+)`")


def main():
    problems, warns = [], []

    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            problems.append(f"THIẾU file bắt buộc: {rel}")

    for rel, marker, msg in VERSION_MARKERS:
        f = ROOT / rel
        if f.exists() and marker not in f.read_text(encoding="utf-8", errors="ignore"):
            (warns if rel.endswith(".yaml") else problems).append(f"{rel}: {msg}")

    # doc: đủ 4 mục + đường dẫn trong doc có tồn tại
    for f in sorted((ROOT / "doc").glob("*.md")):
        text = f.read_text(encoding="utf-8")
        for sec in ("## 1. Purpose", "## 2. IO", "## 3. Success Criteria", "## 4. How to do"):
            if sec not in text:
                problems.append(f"{f.name}: thiếu mục '{sec}'")
        for m in sorted(set(PATH_RE.findall(text))):
            if "*" in m or "{" in m:
                continue
            if not (ROOT / m).exists():
                problems.append(f"{f.name}: nhắc tới `{m}` nhưng file không tồn tại")

    # mọi file src/*.py (trừ __init__) phải được nhắc trong doc/00_index.md
    idx = (ROOT / "doc" / "00_index.md").read_text(encoding="utf-8") if (ROOT / "doc/00_index.md").exists() else ""
    for f in sorted((ROOT / "src").glob("*.py")):
        if f.name != "__init__.py" and f"src/{f.name}" not in idx:
            problems.append(f"doc/00_index.md chưa nhắc tới src/{f.name}")

    print("== Kết quả đã sinh (chỉ thông tin) ==")
    for rel in GENERATED:
        print(f"  [{'có' if (ROOT / rel).exists() else '--'}] {rel}")

    print()
    for w in warns:
        print("CẢNH BÁO:", w)
    for p in problems:
        print("LỖI:", p)
    if problems:
        print(f"\n=> {len(problems)} lỗi. Chép lại file từ bản zip mới nhất rồi chạy lại.")
        sys.exit(1)
    print("=> Cấu trúc và phiên bản code khớp với tài liệu.")


if __name__ == "__main__":
    main()
