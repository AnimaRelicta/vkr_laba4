# Сверка чисел ЛР1

Проверяем: заголовков в test: [255, 276]; классов: [14, 16]. В проекте классов принято 16 (labels.py).

## Папка `C:\Users\spyar\Downloads\laba4\laba4\vkr_laba4\data\lab1\out`

| Определение | train | val | test | все |
| --- | --- | --- | --- | --- |
| строк (наблюдений) | 1440 | 312 | 216 | 1968 |
| уникальных header_raw | 17 | 17 | 12 | 17 |
| уникальных header_text | 17 | 17 | 12 | 17 |
| уникальных header_text (регистр и пробелы не важны) | 17 | 17 | 12 | 17 |
| уникальных пар (класс, header_text) | 17 | 17 | 12 | 17 |
| уникальных троек (класс, header_text, values_sample) | 94 | 52 | 19 | 149 |
| строк без unmapped | 1120 | 248 | 216 | 1584 |
| уникальных header_text без unmapped | 14 | 14 | 12 | 14 |
| уникальных пар (класс, header_text) без unmapped | 14 | 14 | 12 | 14 |

| Классы | train | val | test | все |
| --- | --- | --- | --- | --- |
| классов в gold | 14 | 14 | 12 | 14 |
| классов в gold без unmapped | 13 | 13 | 12 | 13 |

Классы из LABELS без наблюдений: estimate_position, unit. Классы в данных вне LABELS: нет.

## Папка `C:\Users\spyar\Downloads\laba4\laba4\vkr_laba4\data\lab1\out_hard`

| Определение | train | val | test | все |
| --- | --- | --- | --- | --- |
| строк (наблюдений) | 311 | 264 | 255 | 830 |
| уникальных header_text | 211 | 199 | 176 | 451 |
| уникальных header_text (регистр и пробелы не важны) | 195 | 186 | 159 | 405 |
| уникальных пар (класс, header_text) | 226 | 208 | 188 | 486 |
| уникальных троек (класс, header_text, values_sample) | 270 | 246 | 239 | 750 |
| строк без unmapped | 278 | 243 | 243 | 764 |
| уникальных header_text без unmapped | 192 | 184 | 165 | 413 |
| уникальных пар (класс, header_text) без unmapped | 205 | 189 | 177 | 443 |

| Классы | train | val | test | все |
| --- | --- | --- | --- | --- |
| классов в gold | 16 | 16 | 16 | 16 |
| классов в gold без unmapped | 15 | 15 | 15 | 15 |

Классы из LABELS без наблюдений: нет. Классы в данных вне LABELS: нет.

Где эти числа встречаются в текстовых файлах папки:

- `hard_report.json:11: "unit_code": 14,`
- `hard_report.json:17: "vat_rate": 14,`
- `hard_report.json:42: "n": 255,`
- `hard_report.json:65: "combo": 14,`

## Логи LLM `C:\Users\spyar\Downloads\laba4\laba4\vkr_laba4\data\lab1\llm_out_hard`

| Запуск | N | gold | pred (валидных) | pred (всего) | gold вне LABELS |
| --- | --- | --- | --- | --- | --- |
| test__deepseek_deepseek-v4-flash__k0__v0__direct | 255 | 16 | 16 | 17 |  |
| test__google_gemini-3.5-flash-lite__k0__v0__direct | 255 | 16 | 16 | 16 |  |
| test__google_gemma-4-31b-it__k0__v0__direct | 255 | 16 | 16 | 16 |  |
| test__gpt-4o-mini__k0__v0__direct | 255 | 16 | 16 | 16 |  |
| test__openai_gpt-5.6-luna__k0__v0__direct | 255 | 16 | 16 | 17 |  |
| test__qwen_qwen3.8-flash__k0__v0__direct | 255 | 16 | 16 | 17 |  |
| test__qwen_qwen3.8-max-0902__k0__v0__direct | 255 | 16 | 16 | 17 |  |

## Итог: где найдены заявленные числа

- **255**: out_hard/test: строк (наблюдений) = 255; llm_out_hard/test__deepseek_deepseek-v4-flash__k0__v0: n = 255; llm_out_hard/test__google_gemini-3.5-flash-lite__k0__: n = 255; llm_out_hard/test__google_gemma-4-31b-it__k0__v0__dir: n = 255; llm_out_hard/test__gpt-4o-mini__k0__v0__direct: n = 255; llm_out_hard/test__openai_gpt-5.6-luna__k0__v0__direc: n = 255; llm_out_hard/test__qwen_qwen3.8-flash__k0__v0__direct: n = 255; llm_out_hard/test__qwen_qwen3.8-max-0902__k0__v0__dir: n = 255
- **276**: не найдено ни в одном определении
- **14**: out/train: уникальных header_text без unmapped = 14; out/train: уникальных пар (класс, header_text) без unmapped = 14; out/train: классов в gold = 14; out/val: уникальных header_text без unmapped = 14; out/val: уникальных пар (класс, header_text) без unmapped = 14; out/val: классов в gold = 14; out/все: уникальных header_text без unmapped = 14; out/все: уникальных пар (класс, header_text) без unmapped = 14 …
- **16**: out_hard/train: классов в gold = 16; out_hard/val: классов в gold = 16; out_hard/test: классов в gold = 16; out_hard/все: классов в gold = 16; llm_out_hard/test__deepseek_deepseek-v4-flash__k0__v0: gold = 16; llm_out_hard/test__deepseek_deepseek-v4-flash__k0__v0: pred_valid = 16; llm_out_hard/test__google_gemini-3.5-flash-lite__k0__: gold = 16; llm_out_hard/test__google_gemini-3.5-flash-lite__k0__: pred_valid = 16 …

Вывод делается вручную: выберите определение, которое использует отчёт ЛР1, и запишите его в паспорт данных. Если 255 и 276 отличаются, потому что один счёт ведётся по уникальным заголовкам, а другой по строкам или по `out_hard`, в отчёте ЛР4 нужно указывать оба числа с пояснением.
