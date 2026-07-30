#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

PY=python3.13

$PY -m pip install --upgrade pip
# 先装 CPU torch,避免默认拉 CUDA 版
$PY -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
$PY -m pip install -r requirements.txt

exec $PY -m uvicorn app.main:app --host 0.0.0.0 --port 8003