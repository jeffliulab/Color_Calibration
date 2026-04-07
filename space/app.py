"""
Card Calibration — HuggingFace Space Demo

上传包含颜色校准卡的照片，系统自动检测卡片、提取参考色块、
预测目标区域在标准光照下的真实颜色。
"""

import cv2
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage.color import rgb2lab

from model_utils import predict_color

matplotlib.use("Agg")

SPACE_DIR = Path(__file__).parent

MODELS = {
    "XGBoost (Best — ΔE 4.59)": "xgboost",
    "Random Forest": "random_forest",
}


# ── 工具函数 ────────────────────────────────────────────────────────────────

def _contrast_color(rgb):
    """根据背景亮度返回合适的文字颜色。"""
    yiq = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
    return "#222" if yiq >= 128 else "#fff"


def _compute_lab_delta_e(rgb1, rgb2):
    """计算两个 RGB 之间的 CIE76 Lab ΔE。"""
    c1 = np.array(rgb1, dtype=np.float64).reshape(1, 1, 3) / 255.0
    c2 = np.array(rgb2, dtype=np.float64).reshape(1, 1, 3) / 255.0
    lab1 = rgb2lab(c1).reshape(3)
    lab2 = rgb2lab(c2).reshape(3)
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def _parse_rgb(text):
    """解析 'R, G, B' 字符串，返回 tuple 或 None。"""
    if not text or not text.strip():
        return None
    try:
        parts = [int(x.strip()) for x in text.split(",")]
        if len(parts) == 3 and all(0 <= v <= 255 for v in parts):
            return tuple(parts)
    except ValueError:
        pass
    return None


# ── 可视化 ──────────────────────────────────────────────────────────────────

def _make_card_figure(card_crop):
    """将检测到的卡片裁剪区域渲染为 matplotlib figure。"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    card_rgb = cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB)
    ax.imshow(card_rgb)
    ax.set_title("Detected Calibration Card")
    ax.axis("off")
    plt.tight_layout()
    return fig


def _make_color_html(captured_rgb, predicted_rgb, true_rgb=None):
    """生成颜色色块的 HTML 展示。"""
    box_css = (
        "display:inline-flex;flex-direction:column;justify-content:center;"
        "align-items:center;width:150px;height:150px;border-radius:12px;"
        "font-family:'Segoe UI',system-ui,sans-serif;font-weight:700;"
        "font-size:0.9em;box-shadow:0 4px 12px rgba(0,0,0,0.12);"
        "margin:8px;padding:10px;box-sizing:border-box;"
    )

    def _swatch(title, rgb):
        bg = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        fg = _contrast_color(rgb)
        return (
            f'<div style="{box_css}background:{bg};color:{fg};">'
            f"<span>{title}</span>"
            f"<span style='font-size:0.85em;margin-top:6px;'>"
            f"({rgb[0]}, {rgb[1]}, {rgb[2]})</span></div>"
        )

    swatches = [_swatch("Captured", captured_rgb), _swatch("Predicted", predicted_rgb)]
    if true_rgb:
        swatches.append(_swatch("Real", true_rgb))

    return (
        '<div style="display:flex;justify-content:center;flex-wrap:wrap;'
        'gap:12px;margin:16px 0;">' + "".join(swatches) + "</div>"
    )


# ── 核心回调 ────────────────────────────────────────────────────────────────

def run_calibration(image_rgb, model_display, true_rgb_str):
    """处理上传图片，返回检测结果。"""
    if image_rgb is None:
        raise gr.Error("Please upload an image first.")

    model_name = MODELS.get(model_display, "xgboost")
    true_rgb = _parse_rgb(true_rgb_str)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        result = predict_color(image_bgr, model_name=model_name)
    except ValueError as exc:
        raise gr.Error(str(exc))

    captured = result["captured_rgb"]
    predicted = result["predicted_rgb"]
    card_crop = result["card_crop"]

    # 渲染结果
    card_fig = _make_card_figure(card_crop)
    color_html = _make_color_html(captured, predicted, true_rgb)

    # 构建 Summary
    delta_e = _compute_lab_delta_e(captured, predicted)
    lines = [
        f"**Captured (target patch):** RGB{captured}",
        f"**Predicted true color:** RGB{predicted}",
        f"**ΔE (Lab) captured vs predicted:** {delta_e:.2f}",
    ]

    if true_rgb:
        de_cap = _compute_lab_delta_e(captured, true_rgb)
        de_pred = _compute_lab_delta_e(predicted, true_rgb)
        improvement = ((de_cap - de_pred) / de_cap * 100) if de_cap > 0 else 0
        lines += [
            f"**True color:** RGB{true_rgb}",
            f"**ΔE captured vs real:** {de_cap:.2f}",
            f"**ΔE predicted vs real:** {de_pred:.2f}",
        ]
        if improvement > 0:
            lines.append(f"**Accuracy improvement: {improvement:.1f}%**")

    # 质量评级
    if delta_e < 3:
        lines.append("\n*Excellent — professional calibration level.*")
    elif delta_e < 5:
        lines.append("\n*Good — meets commercial printing standards.*")
    elif delta_e < 10:
        lines.append("\n*Acceptable — noticeable under close inspection.*")
    else:
        lines.append("\n*Poor — significant color deviation.*")

    summary = "\n\n".join(lines)
    return card_fig, color_html, summary


# ── Gradio 界面 ─────────────────────────────────────────────────────────────

INTRO = """\
## Card Calibration

Upload a photo containing a **color calibration card** \
(red circle, green triangle, blue pentagon, black box). \
The system detects the card, extracts reference patches, \
and predicts the true color of the target under standard lighting.

**Best model:** XGBoost with Bayesian-optimized hyperparameters — \
**Lab Mean ΔE = 4.59** (commercial printing standard).
"""

with gr.Blocks(
    title="Card Calibration",
    theme=gr.themes.Soft(),
    css=".result-summary { font-size: 1.05em; line-height: 1.7; }",
) as demo:

    gr.Markdown(INTRO)

    # ── 主操作区 ────────────────────────────────────────────────────────
    with gr.Row():
        # 左栏：输入
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Image",
                type="numpy",
                sources=["upload", "clipboard"],
            )
            model_selector = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="XGBoost (Best — ΔE 4.59)",
                label="Model",
            )
            with gr.Accordion("Compare with True Color (optional)", open=False):
                true_rgb_input = gr.Textbox(
                    label="True RGB (R, G, B)",
                    value="",
                    placeholder="e.g. 238, 194, 187",
                    info="Enter known true RGB to see accuracy improvement.",
                )
            run_btn = gr.Button("Run Calibration", variant="primary", size="lg")
            gr.Examples(
                examples=[[str(SPACE_DIR / "examples" / "example1.jpg"), "XGBoost (Best — ΔE 4.59)", ""]],
                inputs=[input_image, model_selector, true_rgb_input],
                label="Try an example",
            )

        # 右栏：输出
        with gr.Column(scale=2):
            card_output = gr.Plot(label="Detected Card")
            color_output = gr.HTML(label="Color Analysis")
            summary_output = gr.Markdown(label="Summary", elem_classes=["result-summary"])

    run_btn.click(
        fn=run_calibration,
        inputs=[input_image, model_selector, true_rgb_input],
        outputs=[card_output, color_output, summary_output],
    )

    # ── 使用说明 & 卡片模板 ─────────────────────────────────────────────
    gr.Markdown("---")
    with gr.Row(equal_height=True):
        with gr.Column():
            gr.Markdown("### How to Use")
            gr.Markdown(
                "1. Print the calibration card (see right).\n"
                "2. Place the card on or next to the object whose color you want to measure.\n"
                "3. Take a photo under any lighting condition.\n"
                "4. Upload the photo above and click **Run Calibration**."
            )
            gr.Image(
                value=str(SPACE_DIR / "usage_photo.png"),
                label="Example Setup",
                interactive=False,
                show_download_button=False,
            )
        with gr.Column():
            gr.Markdown("### Card Template")
            gr.Markdown(
                "Download and print this card. It has four patches:\n"
                "- **Red circle** — red reference\n"
                "- **Green triangle** — green reference\n"
                "- **Blue pentagon** — blue reference\n"
                "- **Black box** — target window (place over the color to measure)"
            )
            gr.Image(
                value=str(SPACE_DIR / "card_template.png"),
                label="Printable Card",
                interactive=False,
                show_download_button=True,
            )

    # ── 页脚 ────────────────────────────────────────────────────────────
    gr.Markdown(
        "---\n"
        "Built with [Gradio](https://gradio.app) · "
        "Models on [HuggingFace Hub](https://huggingface.co/jeffliulab/card-calibration-v1) · "
        "[GitHub](https://github.com/jeffliulab/Color_Calibration)"
    )

if __name__ == "__main__":
    demo.launch()
