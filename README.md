# Shift Scheduler MIP — Gerador de Escala de Turnos

Este projeto gera automaticamente uma **escala semanal de turnos** usando:

- **MIP (Mixed Integer Programming)** com o solver **PuLP**, quando disponível.

O sistema produz:
- **4 arquivos CSV** contendo resultados detalhados.
- **1 dashboard HTML moderno** com visualização completa da escala.

---

## 🚀 Como funciona

O script:

1. Define os **dias da semana**, **turnos** e a **demanda mínima de funcionários** por dia/turno.
2. Tenta resolver usando **otimização inteira linear (MIP)**.
3. Caso o solver não esteja disponível, aplica uma **heurística gulosa**.
4. Gera automaticamente:
   - `empregados_contratados.csv`
   - `escala_por_empregado.csv`
   - `escala_por_turno.csv`
   - `escala_matriz_empregado_dia.csv`
   - `escala_turnos.html` (dashboard)

Todos os arquivos são salvos na pasta **output/**.

---

## 📦 Pré-requisitos

Python 3.8 ou superior.

Instale o PuLP (opcional, mas recomendado):

```bash
pip install pulp pandas
