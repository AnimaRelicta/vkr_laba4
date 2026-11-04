# ВКР: ЛР1 и ЛР4

Проект на `uv` + `pyproject.toml`. Лабораторная 4 построена на данных и коде лабораторной 1 (код ЛР1 не менялся).
Все изменения относительно ЛР1 записаны в **`patches/lab4.patch`** (см. «Патч» ниже).

## ЛР4. Сравнение 8 оптимизаторов

Задача: классификатор «заголовок колонки (+ до 3 значений) -> каноническое поле УПД» (16 классов).
Обучаемая модель по умолчанию - **Qwen3-0.6B + LoRA** (влезает в 8 ГБ RTX 4060); запасной вариант - компактная
модель из ТЗ (хеши n-грамм, менее 1 млн параметров, считается на CPU): `--backend ngram`.
ТЗ и план: см. вложенные файлы задания; рабочие документы проекта: [PROTOCOL.md](PROTOCOL.md), [TODO.md](TODO.md).

### Установка

```bash
# 1. драйвер NVIDIA (>= 570 для CUDA 12.8) и uv: https://docs.astral.sh/uv/
uv sync                       # torch ставится из индекса cu128 (см. pyproject.toml)

# 2. проверка GPU
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3. тесты (тест test_torch_smoke.py сам пропускается, если torch не установлен)
uv run pytest
```

Веса Qwen скачаются с Hugging Face при первом запуске (около 1,2 ГБ в кэш `~/.cache/huggingface`).
Если на Windows `uv sync` поставил CPU-версию torch, `torch.cuda.is_available()` вернёт `False`: проверьте драйвер
и что в `pyproject.toml` на месте блок `[[tool.uv.index]] pytorch-cu128`.

### Данные

Нужна папка `out/` из ЛР1 (результат `prepare_dataset.py`, файл `observations.csv`). Положите её в `data/lab1/out/`
(путь меняется в `configs/protocol.yaml` или ключом `--data`). Без реальных данных конвейер можно проверить на
**синтетических** данных (результаты на них в отчёт не идут):

```bash
uv run lab4 demo-data                         # создаёт data/demo/out
uv run lab4 --data data/demo/out passport
```

### Порядок запуска (этапы плана)

```bash
# этап 2: паспорт данных, split.json, проверка утечек -> artifacts/
uv run lab4 passport

# замер скорости на вашей карте и оценка времени всех этапов
uv run lab4 bench --steps 30 --plan-steps 600

# этап 3: пилот (8 оптимизаторов, seed 0), затем заморозить число шагов и целевой F1
uv run lab4 pilot
uv run lab4 freeze --steps 600 --target 0.90      # после этого configs/*.yaml менять нельзя (protocol.lock.json)

# этап 4: подбор LR (8 x 7 x 2 запуска) и выбор лучшего LR
uv run lab4 tune
uv run lab4 select-lr

# этап 5: основной прогон (8 оптимизаторов x 10 seed, сценарии S0-S5 считаются в каждом запуске)
uv run lab4 main

# этап 6: статистика, сводный балл, гипотезы, графики -> results/<backend>/stats/report.md
uv run lab4 stats

# этап 7: sensitivity для топ-3 по баллу + SGD
uv run lab4 sens

uv run lab4 status                            # сколько запусков выполнено
```

Все команды возобновляемы: уже выполненные запуски (по `run_id`) пропускаются, поэтому прерывать и продолжать можно.
Для проверки на CPU без Qwen добавьте `--backend ngram` (результаты складываются в `results/ngram/`, отдельно от `results/qwen/`).

### Что где лежит

| Путь | Что |
| --- | --- |
| `configs/protocol.yaml, grid.yaml, mutations.yaml, scenarios.yaml` | параметры эксперимента (замораживаются хешем) |
| `src/vkr/lab4/` | код: `data`, `mutations`, `features`, `models`, `optimizers`, `train`, `runner`, `stats`, `report`, `cli` |
| `artifacts/` | `data_passport.md`, `split.json` |
| `results/<backend>/runs.csv` | одна строка на запуск: seed, параметры, метрики, время, память |
| `results/<backend>/logs/` | кривые сходимости: train loss и норма градиента каждые 50 шагов, val F1 каждые 100 |
| `results/<backend>/unstable_runs.csv` | журнал нестабильных запусков с причиной |
| `results/<backend>/probs/` | вероятности классов на test (S0, S3, S4, MAIN) для каскада с LLM |
| `tests/` | `uv run pytest` |

### Патч

`patches/lab4.patch` - все изменения относительно ЛР1 (git-тег `lab1-baseline`). Пересобрать после правок:

```bash
bash scripts/make_patch.sh
```

Применить к чистой копии ЛР1: `git apply patches/lab4.patch`.

---

# ЛР1 (исходное описание)

### Запуск проекта ЛР1

```bash
# синхронизация проекта и сопоставление зависимостей
uv sync

# выгрузка документов из скаченных zip архивов
uv run python extract_upd.py

# получение статистики и подготовка датасета для выполения задания 2 Лабораторной работы 1
uv run python prepare_dataset.py --input upd --out out

# создаем синтетическую вариацию названий колонок дял проверки больших языковых моделей
uv run make_hard_set.py

# запуск zero-shot прогона
uv run python run_llm_benchmark.py --stage zero

# запуск few-shot прогона с 3 примерами
uv run python run_llm_benchmark.py --stage few --shots 3

# запуск для создания сводной статистики после 2 прогонов и определения лучшей модели
uv run python analyze_results.py --dir llm_out_hard
```

### Ссылка на гугл диск с документами и результатами Лабораторной работы 1

https://drive.google.com/drive/folders/1I47B5rXethOOZST8-Pcs0Rs4_sEg01N3?usp=drive_link
