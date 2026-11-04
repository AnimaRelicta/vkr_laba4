#!/usr/bin/env bash
# Собирает patches/lab4.patch: все изменения ЛР4 относительно ЛР1 (git-тег lab1-baseline), включая новые файлы.
# Запуск из корня проекта:  bash scripts/make_patch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
git rev-parse lab1-baseline >/dev/null 2>&1 || { echo "Нет тега lab1-baseline: git tag lab1-baseline <коммит ЛР1>"; exit 1; }
git add -N . >/dev/null 2>&1 || true
git diff --binary lab1-baseline -- . ':(exclude)patches' > patches/lab4.patch
echo "patches/lab4.patch: $(wc -l < patches/lab4.patch) строк, файлов: $(git diff --name-only lab1-baseline -- . ':(exclude)patches' | wc -l)"
