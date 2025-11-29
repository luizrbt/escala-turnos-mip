# turno_avancado_saida_html.py
"""
Gera a escala por solver MIP (PuLP) se disponível; caso contrário usa heurística.
Sempre produz CSVs e um dashboard HTML: escala_turnos.html
"""

import os
from datetime import datetime
from collections import defaultdict

import pandas as pd

# Dados
days = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sab", "Dom"]
shifts = ["M", "T", "N"]
C = {
  ("Seg","M"):8, ("Seg","T"):10, ("Seg","N"):6,
  ("Ter","M"):8, ("Ter","T"):10, ("Ter","N"):6,
  ("Qua","M"):7, ("Qua","T"):9,  ("Qua","N"):6,
  ("Qui","M"):9, ("Qui","T"):11, ("Qui","N"):7,
  ("Sex","M"):10,("Sex","T"):12, ("Sex","N"):8,
  ("Sab","M"):6, ("Sab","T"):8,  ("Sab","N"):5,
  ("Dom","M"):5, ("Dom","T"):7,  ("Dom","N"):5
}

max_shifts_per_week = 5
N_employees = 120  # limite superior para o MIP
employees = list(range(N_employees))

# ---------- Try PuLP MIP ----------
use_mip = False
try:
    import pulp
    use_mip = True
except Exception:
    use_mip = False

def solve_with_pulp():
    """Constrói e resolve o MIP com PuLP. Retorna df_schedule_long, df_hired, df_coverage, df_matrix, info_msg"""
    prob = pulp.LpProblem("Turnos_MinEmpregados", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("y", employees, cat='Binary')
    a = pulp.LpVariable.dicts("a", (employees, days, shifts), cat='Binary')

    # objetivo
    prob += pulp.lpSum(y[i] for i in employees)

    # cobertura mínima
    for d in days:
        for s in shifts:
            prob += pulp.lpSum(a[i][d][s] for i in employees) >= C[(d,s)]

    # só trabalha se contratado
    for i in employees:
        for d in days:
            for s in shifts:
                prob += a[i][d][s] <= y[i]

    # max 1 turno por dia
    for i in employees:
        for d in days:
            prob += pulp.lpSum(a[i][d][s] for s in shifts) <= 1

    # Max turnos por semana
    for i in employees:
        prob += pulp.lpSum(a[i][d][s] for d in days for s in shifts) <= max_shifts_per_week

    # Solve
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
    except Exception as e:
        raise RuntimeError("Solver falhou: " + str(e))

    # Coletar a solução
    hired = [i for i in employees if pulp.value(y[i]) is not None and pulp.value(y[i]) > 0.5]
    records = []
    for i in hired:
        for d in days:
            for s in shifts:
                if pulp.value(a[i][d][s]) is not None and pulp.value(a[i][d][s]) > 0.5:
                    records.append([i, d, s])

    df_schedule_long = pd.DataFrame(records, columns=["Empregado", "Dia", "Turno"])
    df_hired = pd.DataFrame({"Empregado": hired})

    # coverage
    records2 = []
    for d in days:
        for s in shifts:
            alocados = df_schedule_long[(df_schedule_long['Dia']==d) & (df_schedule_long['Turno']==s)]['Empregado'].tolist()
            records2.append([d, s, len(alocados), alocados])
    df_coverage = pd.DataFrame(records2, columns=["Dia","Turno","Funcionários Alocados","Lista_Empregados"])

    # Matriz empregado x dia
    df_matrix = df_schedule_long.pivot_table(index="Empregado", columns="Dia", values="Turno", aggfunc='first')
    for d in days:
        if d not in df_matrix.columns:
            df_matrix[d] = ""
    df_matrix = df_matrix[days]

    info_msg = "Solução via MIP (PuLP). Status: " + pulp.LpStatus[prob.status]
    return df_schedule_long, df_hired, df_coverage, df_matrix, info_msg

# ---------- Heurística (fallback) ----------
def solve_with_heuristic():
    """Heurística gulosa que preenche turno a turno reaproveitando funcionários com menor carga."""
    employees_list = []  # cada item: {'assignments':{day:shift}, 'count':int}
    def find_existing_employee(day):
        candidates = [(i, e['count']) for i, e in enumerate(employees_list) if day not in e['assignments'] and e['count'] < max_shifts_per_week]
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    for d in days:
        for s in shifts:
            req = C[(d,s)]
            for _ in range(req):
                idx = find_existing_employee(d)
                if idx is None:
                    new_id = len(employees_list)
                    employees_list.append({'assignments': {}, 'count': 0})
                    idx = new_id
                employees_list[idx]['assignments'][d] = s
                employees_list[idx]['count'] += 1

    records = []
    for i, e in enumerate(employees_list):
        for d, s in e['assignments'].items():
            records.append([i, d, s])
    df_schedule_long = pd.DataFrame(records, columns=["Empregado", "Dia", "Turno"])
    df_hired = pd.DataFrame({"Empregado": list(range(len(employees_list)))})
    # coverage
    records2 = []
    for d in days:
        for s in shifts:
            alocados = df_schedule_long[(df_schedule_long['Dia']==d) & (df_schedule_long['Turno']==s)]['Empregado'].tolist()
            records2.append([d, s, len(alocados), alocados])
    df_coverage = pd.DataFrame(records2, columns=["Dia","Turno","Funcionários Alocados","Lista_Empregados"])
    df_matrix = df_schedule_long.pivot_table(index="Empregado", columns="Dia", values="Turno", aggfunc='first')
    for d in days:
        if d not in df_matrix.columns:
            df_matrix[d] = ""
    df_matrix = df_matrix[days]
    info_msg = "Solução via heurística gulosa (fallback). Pode não ser ótima."
    return df_schedule_long, df_hired, df_coverage, df_matrix, info_msg

# ---------- Run appropriate solver ----------
if use_mip:
    try:
        df_schedule_long, df_hired, df_coverage, df_matrix, info_msg = solve_with_pulp()
    except Exception as e:
        # fallback if solver error
        df_schedule_long, df_hired, df_coverage, df_matrix, info_msg = solve_with_heuristic()
        info_msg += f" (MIP tentou e falhou: {e})"
else:
    df_schedule_long, df_hired, df_coverage, df_matrix, info_msg = solve_with_heuristic()

# ---------- SALVAR CSVs ----------
os.makedirs("output", exist_ok=True)
csv_hired = os.path.join("output", "empregados_contratados.csv")
csv_schedule_emp = os.path.join("output", "escala_por_empregado.csv")
csv_coverage = os.path.join("output", "escala_por_turno.csv")
csv_matrix = os.path.join("output", "escala_matriz_empregado_dia.csv")

df_hired.to_csv(csv_hired, index=False)
df_schedule_long.to_csv(csv_schedule_emp, index=False)
df_coverage.to_csv(csv_coverage, index=False)
df_matrix.to_csv(csv_matrix)

# ---------- GERAR HTML ----------
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
total_hired = len(df_hired)
total_assigned_shifts = len(df_schedule_long)
coverage_total = sum(C.values())

def build_html():
    html = f"""<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <title>Escala de Turnos - Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body {{ font-family: Inter, Roboto, Arial, sans-serif; background:#0f172a; color:#e6eef8; margin:0; padding:20px; }}
    .container {{ max-width:1200px; margin:0 auto; }}
    h1 {{ margin:0 0 10px 0; font-size:24px; color:#fff; }}
    .meta {{ color:#9fb0d6; margin-bottom:20px; }}
    .cards {{ display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap; }}
    .card {{ background:linear-gradient(135deg,#0b1220,#122433); padding:16px; border-radius:12px; box-shadow:0 6px 18px rgba(2,6,23,0.6); min-width:180px; }}
    .card h2 {{ margin:0; font-size:20px; color:#9fe7c4; }}
    table {{ border-collapse:collapse; width:100%; margin-bottom:20px; background:transparent; }}
    th, td {{ border:1px solid rgba(255,255,255,0.06); padding:8px 10px; text-align:center; }}
    th {{ background:rgba(255,255,255,0.03); color:#bfe0ff; }}
    td {{ color:#e6eef8; }}
    .shift-M {{ color:#1fb3ff; font-weight:700; }}
    .shift-T {{ color:#ffd36b; font-weight:700; }}
    .shift-N {{ color:#d5a6ff; font-weight:700; }}
    .small {{ font-size:13px; color:#9fb0d6; }}
    .downloads {{ margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; }}
    .btn {{ padding:8px 12px; background:#1e293b; color:#cde9ff; border-radius:8px; text-decoration:none; border:1px solid rgba(255,255,255,0.04); }}
    .btn:hover {{ transform:translateY(-2px); box-shadow:0 8px 20px rgba(2,6,23,0.6); }}
    .emp-row:hover {{ background:rgba(255,255,255,0.02); }}
    .legend {{ display:flex; gap:12px; align-items:center; margin-bottom:8px; }}
    .legend span {{ display:inline-block; width:12px; height:12px; border-radius:3px; }}
    .leg-M {{ background:#1fb3ff; }} .leg-T {{ background:#ffd36b; }} .leg-N {{ background:#d5a6ff; }}
    .footer {{ color:#8ea9d4; font-size:13px; margin-top:24px; }}
    @media (max-width:900px) {{ .cards {{ flex-direction:column; }} }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Escala de Turnos — Dashboard</h1>
    <div class="meta small">Gerado em: {now_str} • {info_msg} • max_shifts_per_week = {max_shifts_per_week}</div>

    <div class="cards">
      <div class="card"><div class="small">Empregados contratados</div><h2>{total_hired}</h2><div class="small">Funcionários distintos alocados</div></div>
      <div class="card"><div class="small">Turnos atribuídos</div><h2>{total_assigned_shifts}</h2><div class="small">Turnos semanais preenchidos</div></div>
      <div class="card"><div class="small">Cobertura mínima total</div><h2>{coverage_total}</h2><div class="small">Soma das exigências por turno</div></div>
    </div>

    <div style="display:flex; justify-content:space-between; gap:20px; flex-wrap:wrap;">
      <div style="flex:1; min-width:320px;">
        <h3 style="margin-bottom:6px;">Cobertura por Dia e Turno</h3>
        <div class="legend">
          <div class="leg-M"></div><div class="small">M (Manhã)</div>
          <div class="leg-T"></div><div class="small">T (Tarde)</div>
          <div class="leg-N"></div><div class="small">N (Noite)</div>
        </div>
        <table><thead><tr><th>Dia</th><th>Turno</th><th>Alocados</th><th>Exigido</th></tr></thead><tbody>
"""
    for _, row in df_coverage.iterrows():
        dia = row['Dia']; turno = row['Turno']; alocados = int(row['Funcionários Alocados']); exigido = C[(dia, turno)]
        cls = f"shift-{turno}"
        html += f"<tr><td>{dia}</td><td class='{cls}'>{turno}</td><td>{alocados}</td><td>{exigido}</td></tr>\n"

    html += f"""
          </tbody></table>
        <div class="downloads">
          <a class="btn" href="{csv_hired}" download>CSV Empregados</a>
          <a class="btn" href="{csv_schedule_emp}" download>CSV Escala por Empregado</a>
          <a class="btn" href="{csv_coverage}" download>CSV Escala por Turno</a>
          <a class="btn" href="{csv_matrix}" download>CSV Matriz Empregado×Dia</a>
        </div>
      </div>

      <div style="flex:1.4; min-width:420px;">
        <h3>Escala: tabela Empregado × Dia</h3>
        <div style="overflow:auto; max-height:520px; border-radius:8px; padding:6px; background:linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.00));">
          <table><thead><tr><th>Emp</th>"""
    for d in days:
        html += f"<th>{d}</th>"
    html += "</tr></thead><tbody>"

    for emp in df_matrix.index.sort_values():
        html += f"<tr class='emp-row'><td>{emp}</td>"
        for d in days:
            val = df_matrix.at[emp, d] if d in df_matrix.columns else ""
            if pd.isna(val): val = ""
            cls = ""
            if val == "M": cls = "shift-M"
            elif val == "T": cls = "shift-T"
            elif val == "N": cls = "shift-N"
            display_val = val if val else ""
            html += f"<td class='{cls}'>{display_val}</td>"
        html += "</tr>"

    html += f"""
            </tbody></table>
        </div>
      </div>
    </div>

    <div class="footer">
      Observações: cada empregado tem no máximo {max_shifts_per_week} turnos/semana. {info_msg}
    </div>

  </div>
</body>
</html>
"""
    return html

html_content = build_html()
html_path = os.path.join("output", "escala_turnos.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

# ---------- RESUMO (impressão curta) ----------
print("Dashboard gerado:", html_path)
print("CSV: ", csv_hired, csv_schedule_emp, csv_coverage, csv_matrix)
print("Total empregados distintos alocados:", total_hired)
print("Total de turnos preenchidos:", total_assigned_shifts)
print("Modo usado:", info_msg)
