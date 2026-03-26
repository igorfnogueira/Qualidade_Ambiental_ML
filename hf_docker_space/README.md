---
title: Qualidade Ambiental (IA)
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

Interface web + API FastAPI para classificar **qualidade ambiental** com modelo treinado (Random Forest). Envie variáveis ambientais e receba a classe prevista com probabilidades.

**Uso local (Docker):** na raiz deste repositório, `docker build -t qa-ml .` e `docker run -p 7860:7860 qa-ml`, depois abra `http://127.0.0.1:7860`.
