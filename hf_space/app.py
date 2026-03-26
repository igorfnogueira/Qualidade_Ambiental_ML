"""Interface Gradio para inferência de Qualidade_Ambiental."""

from __future__ import annotations

import os

import gradio as gr

from inference import FEATURE_COLUMNS, predict_from_tuple, validate_inputs

DEFAULTS = {
    "Temperatura": 22.0,
    "Umidade": 50.0,
    "CO2": 1200.0,
    "CO": 25.0,
    "Pressao_Atm": 1013.0,
    "NO2": 40.0,
    "SO2": 30.0,
    "O3": 50.0,
}


def _run(*vals: float | None) -> tuple[str, str]:
    try:
        errors, warnings = validate_inputs(vals)
        if errors:
            details = "Erros de validação (corrija para prever):\n" + "\n".join(errors)
            return "Erro: entradas inválidas", details

        classe, extra = predict_from_tuple(vals)
        parts: list[str] = []
        if warnings:
            parts.append("Avisos (valores incomuns, mas a predição foi executada):")
            parts.extend(warnings)
            parts.append("")
        parts.append(extra or "(sem predict_proba)")
        return classe, "\n".join(parts).strip()
    except Exception as e:
        return f"Erro: {e}", ""


def build_demo() -> gr.Blocks:
    css = r"""
    .gradio-container {
        background: url("file=assets/bg.png") center / cover no-repeat fixed !important;
        min-height: 100vh;
    }
    .qa_shell {
        max-width: 1100px;
        margin: 0 auto;
        padding: 28px 18px 40px;
    }
    .qa_card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 18px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.10);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        padding: 18px 18px 14px;
    }
    .qa_title h1 { margin-bottom: 8px; }
    .qa_notice {
        margin-top: 10px;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1px solid rgba(34, 197, 94, 0.35);
        background: rgba(220, 252, 231, 0.75);
        font-weight: 600;
    }
    .qa_btn button {
        height: 52px !important;
        border-radius: 16px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, rgba(34,197,94,1) 0%, rgba(16,185,129,1) 100%) !important;
        border: 0 !important;
    }
    """

    with gr.Blocks(title="Qualidade Ambiental", css=css) as demo:
        with gr.Column(elem_classes=["qa_shell"]):
            with gr.Column(elem_classes=["qa_title"]):
                gr.Markdown(
                    "# Qualidade ambiental (classificação)\n\n"
                    "Informe as mesmas variáveis do dataset de treino. A saída é a classe "
                    "**Qualidade_Ambiental** prevista pelo modelo exportado."
                )
                gr.Markdown(
                    "**Este conteúdo é destinado apenas\n"
                    "para fins educacionais. Os dados exibidos são ilustrativos e podem não\n"
                    "corresponder a situações reais.**",
                    elem_classes=["qa_notice"],
                )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes=["qa_card"]):
                    gr.Markdown("### Informe as variáveis ambientais")
                    inputs = [
                        gr.Number(label=name, value=DEFAULTS.get(name))
                        for name in FEATURE_COLUMNS
                    ]
                    btn = gr.Button("Prever", elem_classes=["qa_btn"])

                with gr.Column(scale=1, elem_classes=["qa_card"]):
                    gr.Markdown("### Qualidade_Ambiental prevista")
                    out_class = gr.Textbox(label="Qualidade_Ambiental (prevista)")
                    out_proba = gr.Textbox(label="Detalhes", lines=14)

            btn.click(fn=_run, inputs=inputs, outputs=[out_class, out_proba])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
