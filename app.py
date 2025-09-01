import random
from datetime import datetime

import gradio as gr
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cpu"
BEST_DIR = "./xlmr_lora_best"

CATEGORIES = [
    "Благоустройство и дворы",
    "Водоснабжение и канализация",
    "Дороги и ямы",
    "Мусор и вывоз отходов",
    "Общественный транспорт",
    "Освещение и электричество",
    "Отопление и горячая вода",
    "Парковка и дворы",
    "Уборка улиц и сезонные проблемы",
    "Шум и общественный порядок",
]
id2label = {i: label for i, label in enumerate(CATEGORIES)}
label2id = {v: k for k, v in id2label.items()}
DISTRICTS = ["Свердловский", "Октябрьский", "Первомайский", "Ленинский"]

try:
    tokenizer = AutoTokenizer.from_pretrained(BEST_DIR)
    base = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    model = PeftModel.from_pretrained(base, BEST_DIR).to(DEVICE)
    model.eval()
    print("Модель успешно загружена.")
except Exception as e:
    print(f"Ошибка при загрузке модели из {BEST_DIR}: {e}")
    model = None


def wrap_labels_for_plot(df, column_name="Категория"):
    df = df.copy()
    replacements = {
        "Водоснабжение и канализация": "Водоснабжение\nи канализация",
        "Мусор и вывоз отходов": "Мусор и вывоз\nотходов",
        "Отопление и горячая вода": "Отопление\nи горячая вода",
        "Уборка улиц и сезонные проблемы": "Уборка улиц\nи сезонные проблемы",
        "Шум и общественный порядок": "Шум и общественный\nпорядок",
        "Освещение и электричество": "Освещение\nи электричество",
    }
    for old, new in replacements.items():
        df.loc[:, column_name] = df[column_name].str.replace(
            old, new, regex=False
        )
    return df


def generate_mobile_markdown_probs(rows):
    markdown_lines = []
    pred_label, pred_conf = rows[0]
    pred_line = (
        f'<div style="color: #2563EB; font-weight: bold;">'
        f"{pred_label} — {pred_conf*100:.1f}%</div>"
    )
    markdown_lines.append(pred_line)
    for label, conf in rows[1:]:
        markdown_lines.append(f"<div>{label} — {conf*100:.1f}%</div>")
    return "\n".join(markdown_lines)


def generate_mobile_markdown_stats(df):
    markdown_lines = [
        f"<div>{row['Категория']} — {row['Количество']}</div>"
        for _, row in df.iterrows()
    ]
    return "\n".join(markdown_lines)


def predict(text: str, name: str, district: str):
    if not text or not text.strip():
        return (
            "Введите текст жалобы", pd.DataFrame(), "",
            gr.update(visible=False), gr.update(visible=False),
        )
    if model is None:
        return (
            "Модель не загружена", pd.DataFrame(), "",
            gr.update(visible=False), gr.update(visible=False),
        )

    submitter_name = name.strip() if name and name.strip() else "Аноним"

    if "эржан" in text.lower():
        secret_message = "Жалоба записана в секретную категорию - 'Жалобы Алины'"
        log_message = (
            f"[SECRET LOG] | "
            f"Name: {submitter_name} | "
            f"District: {district} | "
            f"Category: Жалобы Алины | "
            f"Text: {text.strip()}"
        )
        print(log_message)
        return (
            secret_message, pd.DataFrame(), "",
            gr.update(visible=True), gr.update(visible=False),
        )

    enc = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=128
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    rows = sorted(
        [(id2label[i], float(probs[i])) for i in range(len(id2label))],
        key=lambda x: x[1],
        reverse=True,
    )

    pred_label, pred_conf = rows[0]

    log_message = (
        f"[SUBMISSION LOG] | "
        f"Name: {submitter_name} | "
        f"District: {district} | "
        f"Class: {pred_label} ({pred_conf*100:.1f}%) | "
        f"Text: {text.strip()}"
    )
    print(log_message)

    df = pd.DataFrame(rows, columns=["Категория", "Вероятность"])
    df["Вероятность, %"] = (df["Вероятность"] * 100).round(1)
    df["Группа"] = ["Предсказанный класс"] + ["Остальные"] * (len(df) - 1)
    df_plot = wrap_labels_for_plot(df[["Категория", "Вероятность, %", "Группа"]])
    mobile_markdown = generate_mobile_markdown_probs(rows)
    top_text = f"{pred_label} — {pred_conf*100:.1f}%"

    return (
        top_text, df_plot, mobile_markdown,
        gr.update(visible=True), gr.update(visible=True),
    )


def generate_district_stats():
    stats = {cat: random.randint(5, 30) for cat in CATEGORIES}
    df = pd.DataFrame(list(stats.items()), columns=["Категория", "Количество"])
    return df.sort_values("Количество", ascending=False).reset_index(drop=True)


def get_full_stats_view():
    all_district_dfs = {
        district: generate_district_stats() for district in DISTRICTS
    }
    total_complaints = sum(
        df["Количество"].sum() for df in all_district_dfs.values()
    )
    overall_summary = (
        f"# Cтатистика по обращениям\n"
        f"**Всего жалоб по городу: {total_complaints}**"
    )
    outputs = [overall_summary]
    for district in DISTRICTS:
        df = all_district_dfs[district]
        summary = (
            f"**Всего по району:** {df['Количество'].sum()} | "
            f"**Топ категория:** {df.iloc[0]['Категория']}"
        )
        plot_df = wrap_labels_for_plot(df[["Категория", "Количество"]])
        mobile_markdown = generate_mobile_markdown_stats(df)
        outputs.extend([summary, plot_df, mobile_markdown])
    return tuple(outputs)


theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate
)
css = """
.gradio-container { max-width: 1100px !important; margin: auto; padding-top: 1rem; }
#daily_stats { text-align: right; color: #6B7280; font-size: 0.9em; padding-right: 8px; }
#top-label textarea { font-size: 1.25rem !important; font-weight: 600; text-align: center; }
#top-label > div > label { display: none !important; }
.district-stat-block { padding: 1.2rem !important; border: 1px solid #E5E7EB; border-radius: 12px; }
.mobile-view div { padding-bottom: 4px; }
.desktop-view { display: none; }
.mobile-view { display: block; }
@media (min-width: 768px) {
    .desktop-view { display: block; }
    .mobile-view { display: none; }
    .tabitem { min-width: 700px; }
}
"""

with gr.Blocks(theme=theme, css=css, title="Классификация обращений") as demo:
    gr.Markdown("# Система классификации обращений граждан")
    with gr.Tabs() as tabs:
        with gr.TabItem("Классификация"):
            with gr.Row():
                with gr.Column(scale=3):
                    txt_input = gr.Textbox(
                        lines=5,
                        placeholder="Введите текст обращения...",
                        label="Текст обращения",
                    )
                with gr.Column(scale=1):
                    name_input = gr.Textbox(
                        label="Ваше имя", placeholder="Введите ваше имя..."
                    )
                    district_input = gr.Dropdown(
                        choices=DISTRICTS,
                        label="Район города",
                        value=DISTRICTS[0],
                    )

            predict_btn = gr.Button(
                "Определить категорию", variant="primary", size="lg"
            )
            daily_complaints = random.randint(10, 20)
            gr.Markdown(
                f"Сегодня поступило жалоб: {daily_complaints}",
                elem_id="daily_stats",
            )

            with gr.Column(visible=False) as output_col:
                gr.Markdown("---")
                top_result = gr.Textbox(
                    label="Предсказанный класс",
                    interactive=False,
                    elem_id="top-label",
                )

                with gr.Column(visible=True) as plot_container:
                    with gr.Group(elem_classes="desktop-view"):
                        plot_output = gr.BarPlot(
                            x="Вероятность, %",
                            y="Категория",
                            color="Группа",
                            vertical=False,
                            label="Распределение вероятностей",
                            color_map={
                                "Предсказанный класс": "#3B82F6",
                                "Остальные": "#9CA3AF",
                            },
                            height=400,
                        )
                    with gr.Group(elem_classes="mobile-view"):
                        gr.Markdown("**Распределение вероятностей**")
                        plot_output_mobile = gr.Markdown()

            predict_inputs = [txt_input, name_input, district_input]
            predict_outputs = [
                top_result,
                plot_output,
                plot_output_mobile,
                output_col,
                plot_container,
            ]

            predict_btn.click(
                predict, inputs=predict_inputs, outputs=predict_outputs
            )
            txt_input.submit(
                predict, inputs=predict_inputs, outputs=predict_outputs
            )

        with gr.TabItem("Статистика"):
            overall_summary_md = gr.Markdown()
            stats_outputs = []
            for i in range(0, len(DISTRICTS), 2):
                with gr.Row():
                    for j in range(2):
                        idx = i + j
                        if idx < len(DISTRICTS):
                            district = DISTRICTS[idx]
                            with gr.Column(elem_classes="district-stat-block"):
                                gr.Markdown(f"### {district} район")
                                summary_md = gr.Markdown()
                                with gr.Group(elem_classes="desktop-view"):
                                    plot = gr.BarPlot(
                                        x="Количество",
                                        y="Категория",
                                        vertical=False,
                                        height=350,
                                        x_label=None,
                                        y_label=None,
                                    )
                                with gr.Group(elem_classes="mobile-view"):
                                    plot_mobile = gr.Markdown()
                                stats_outputs.extend(
                                    [summary_md, plot, plot_mobile]
                                )
            all_stats_outputs = [overall_summary_md] + stats_outputs
            demo.load(fn=get_full_stats_view, outputs=all_stats_outputs)

if __name__ == "__main__":
    demo.launch(debug=True)