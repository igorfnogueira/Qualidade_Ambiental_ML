const FEATURE_COLUMNS = [
  "Temperatura",
  "Umidade",
  "CO2",
  "CO",
  "Pressao_Atm",
  "NO2",
  "SO2",
  "O3",
];

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function parseNumber(v) {
  if (v === "" || v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function buildPayload(form) {
  const payload = {};
  for (const k of FEATURE_COLUMNS) {
    const input = form.querySelector(`[name="${k}"]`);
    payload[k] = parseNumber(input?.value ?? "");
  }
  return payload;
}

function formatDetails({ warnings, details, errors }) {
  const parts = [];
  if (errors && errors.length) {
    parts.push("Erros de validação (corrija para prever):");
    parts.push(...errors);
    return parts.join("\n");
  }
  if (warnings && warnings.length) {
    parts.push("Avisos (valores incomuns, mas a predição foi executada):");
    parts.push(...warnings);
    parts.push("");
  }
  parts.push(details || "(sem predict_proba)");
  return parts.join("\n").trim();
}

async function onSubmit(e) {
  e.preventDefault();
  const form = e.currentTarget;
  const btn = document.getElementById("btn-predict");
  const payload = buildPayload(form);

  setText("predicted", "Processando…");
  setText("details", "Enviando dados para o modelo…");
  if (btn) btn.disabled = true;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      setText("predicted", "—");
      setText(
        "details",
        formatDetails({
          errors: data?.errors || ["- Falha na validação. Verifique os campos."],
          warnings: data?.warnings || [],
          details: "",
        })
      );
      return;
    }

    setText("predicted", data?.qualidade_ambiental ?? "—");
    setText(
      "details",
      formatDetails({
        warnings: data?.warnings || [],
        details: data?.details || "",
        errors: [],
      })
    );
  } catch (err) {
    setText("predicted", "—");
    setText("details", `Erro ao chamar a API: ${err}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function onReset() {
  const form = document.getElementById("predict-form");
  if (!form) return;
  for (const k of FEATURE_COLUMNS) {
    const input = form.querySelector(`[name="${k}"]`);
    if (input) input.value = "";
  }
  setText("predicted", "—");
  setText("details", "Preencha os campos e clique em “Prever”.");
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const reset = document.getElementById("btn-reset");
  if (form) form.addEventListener("submit", onSubmit);
  if (reset) reset.addEventListener("click", onReset);
});

