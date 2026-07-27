import numpy as np
import pandas as pd

from config import config
from src.services.simulation.sim_data import SCALE_CONFIG as SC_SIM, generate_dataset
from src.models.skip_tracing import SCALE_CONFIG as SC_ST, generate as gen_st
from src.models.identity_resolution import SCALE_CONFIG as SC_IR, generate as gen_irj

INK = "#0B0E11"; PANEL = "#12161C"; LINE = "#232B34"; TEXT = "#D7DEE6"; MUTED = "#7C8A99"
GOOD = "#33D6A6"; BAD = "#F0654A"; ACCENT = "#4FA8FF"; ACCENT2 = "#C792EA"
SCALES = ["SMALL", "MEDIUM", "LARGE"]


def kpi(label, value, sub=""):
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>' \
           f'<div class="kpi-sub">{sub}</div></div>'


def grouped_bar_chart(groups, series, w=740, h=260, fmt="{:.1f}", unit=""):
    pad_l, pad_r, pad_t, pad_b = 26, 20, 24, 34
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    vmax = max(g["values"][k] for g in groups for k in g["values"]) * 1.3 or 1
    n_groups = len(groups)
    n_series = len(series)
    slot = plot_w / n_groups
    bw = slot * 0.72 / n_series
    parts = []
    base_y = pad_t + plot_h
    parts.append(f'<line x1="{pad_l}" x2="{w-pad_r}" y1="{base_y}" y2="{base_y}" stroke="{LINE}" stroke-width="1"/>')
    for gi, g in enumerate(groups):
        gx = pad_l + gi * slot + slot * 0.14
        for si, (key, color, _) in enumerate(series):
            v = g["values"][key]
            bh = max(v, 0) / vmax * plot_h
            bx = gx + si * bw
            by = base_y - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw*0.82:.1f}" height="{bh:.1f}" fill="{color}" rx="2"/>')
            parts.append(f'<text x="{bx+bw*0.41:.1f}" y="{by-6:.1f}" text-anchor="middle" font-size="10.5" '
                         f'font-weight="600" fill="{color}" font-family="IBM Plex Mono,monospace">{fmt.format(v)}{unit}</text>')
        parts.append(f'<text x="{pad_l+gi*slot+slot/2:.1f}" y="{h-10}" text-anchor="middle" font-size="12.5" '
                     f'fill="{TEXT}" font-family="IBM Plex Mono,monospace" font-weight="600">{g["label"]}</text>')
    legend = "".join(f'<span><i style="background:{c}"></i>{lbl}</span>' for _, c, lbl in series)
    return f'<svg viewBox="0 0 {w} {h}" width="100%">{"".join(parts)}</svg><div class="legend">{legend}</div>'


def table(headers, rows):
    th = "".join(f"<th>{x}</th>" for x in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="panel-table"><table><tr>{th}</tr>{tr}</table></div>'


def section_head(num, title, sub, desc):
    return f'''<div class="prob-head"><span class="prob-num">{num}</span>
      <div><div class="prob-title">{title}</div><div class="prob-sub">{sub}</div>
      <div class="prob-desc">{desc}</div></div></div>'''


def note(text, cls="good"):
    return f'<div class="note {cls}"><span class="lbl">Insight</span>{text}</div>'


sim_stats = {}
for name, cfg in SC_SIM.items():
    customers, edges = generate_dataset(**cfg)
    seg = customers.set_index("id")["_true_segment"]
    default_by_seg = customers.groupby("_true_segment")["default"].mean()
    n_comm = default_by_seg.shape[0]
    n_risk_seg = max(1, int(0.12 * n_comm))
    risky = default_by_seg.sort_values(ascending=False).head(n_risk_seg).index
    is_risky_cust = customers["_true_segment"].isin(risky)
    within = (seg.loc[edges["src"]].to_numpy() == seg.loc[edges["tgt"]].to_numpy()).mean()
    deg = pd.concat([edges["src"], edges["tgt"]]).value_counts()
    avg_deg = deg.reindex(customers["id"], fill_value=0).mean()
    corr_debt_app = customers["debt_amount"].corr(customers["application_amount"])
    sim_stats[name] = dict(
        n=len(customers), n_comm=n_comm, n_risk_seg=n_risk_seg,
        default_overall=customers["default"].mean(),
        default_risky=customers.loc[is_risky_cust, "default"].mean(),
        default_normal=customers.loc[~is_risky_cust, "default"].mean(),
        within_edge_pct=within, avg_degree=avg_deg, n_edges=len(edges),
        debt_median=customers["debt_amount"].median(),
        debt_p95=customers["debt_amount"].quantile(0.95),
        overdue_mean=customers["overdue_days"].mean(),
        phone_active_mean=customers["phone_active_ratio"].mean(),
        address_active_rate=customers["address_active"].mean(),
        corr_debt_app=corr_debt_app,
    )

st_stats = {}
for name, cfg in SC_ST.items():
    df = gen_st(**cfg)
    lc = df[df.lost_contact]
    st_stats[name] = dict(
        n=cfg["n_customers"], lost_rate=df["lost_contact"].mean(),
        reach_base=lc["baseline_reach"].mean(), reach_graph=lc["graph_reach"].mean(),
        via_guarantor=lc["reach_via_guarantor"].mean(), via_person=lc["reach_via_person"].mean(),
        via_address=lc["reach_via_shared_address"].mean(),
        overall_address_active=df["address_active"].mean(),
    )

ir_stats = {}
for name, cfg in SC_IR.items():
    df = gen_irj(**cfg)
    dup_groups = df.groupby("true_id").filter(lambda g: len(g) > 1)
    n_dup_persons = dup_groups["true_id"].nunique()
    same_phone = dup_groups.groupby("true_id")["phone"].nunique().eq(1).mean()
    same_addr = dup_groups.groupby("true_id")["address"].nunique().eq(1).mean()
    same_nid = dup_groups.groupby("true_id")["national_id"].nunique().eq(1).mean()
    phone_collision = df.groupby("phone")["true_id"].nunique().gt(1).sum()
    addr_collision = df.groupby("address")["true_id"].nunique().gt(1).sum()
    ir_stats[name] = dict(
        n_persons=cfg["n_persons"], n_records=len(df), n_dup_persons=n_dup_persons,
        dup_rate=n_dup_persons / cfg["n_persons"],
        same_phone_rate=same_phone, same_addr_rate=same_addr, same_nid_rate=same_nid,
        phone_collision=phone_collision, addr_collision=addr_collision,
    )

sim_rows = [(k, sim_stats[k]["n"], sim_stats[k]["n_comm"], sim_stats[k]["n_risk_seg"],
             f'{sim_stats[k]["avg_degree"]:.1f}', f'{sim_stats[k]["within_edge_pct"]:.1%}',
             f'{sim_stats[k]["default_overall"]:.1%}') for k in SCALES]

sim_risk_bar = grouped_bar_chart(
    [dict(label=k, values=dict(risky=sim_stats[k]["default_risky"] * 100,
                                normal=sim_stats[k]["default_normal"] * 100)) for k in SCALES],
    [("risky", BAD, "Cộng đồng rủi ro cao"), ("normal", ACCENT, "Cộng đồng thường")],
    fmt="{:.1f}", unit="%")

section_sim = f'''<section>
{section_head("01", "sim_data.py — Dữ liệu lõi (Supply Chain & Segmentation)",
  "Khách hàng + đồ thị quan hệ giao dịch, sinh theo cấu trúc cộng đồng",
  "Mỗi khách hàng thuộc 1 trong N cộng đồng; cạnh đồ thị được sinh thiên lệch để phần lớn nằm trong cùng "
  "cộng đồng (mô phỏng homophily). ~12% cộng đồng có rủi ro nội tại cao hơn hẳn — số liệu dưới đây được tính "
  "trực tiếp từ dữ liệu sinh ra ở mỗi lần chạy, không phải tham số khai báo sẵn.")}
{table(["Quy mô", "n khách hàng", "n cộng đồng", "n cộng đồng rủi ro cao", "Bậc trung bình (degree)",
        "% cạnh trong cùng cộng đồng", "Tỷ lệ default"], sim_rows)}
<div class="crit-card">
  <div class="crit-name">Default rate: cộng đồng rủi ro cao vs cộng đồng thường</div>
  {sim_risk_bar}
</div>
{note(f'Ở mọi quy mô, nhóm cộng đồng rủi ro cao có tỷ lệ default cao gấp '
      f'{sim_stats["LARGE"]["default_risky"]/max(sim_stats["LARGE"]["default_normal"],1e-9):.1f} lần nhóm còn lại '
      f'(Large) — đây chính là tín hiệu mà feature đồ thị (degree, community, distance-to-hub) kỳ vọng bắt được. '
      f'Tương quan debt_amount ↔ application_amount ≈ {sim_stats["LARGE"]["corr_debt_app"]:.2f} — cẩn thận '
      f'multicollinearity khi diễn giải feature importance. Tỷ lệ cạnh nằm trong cùng cộng đồng luôn quanh '
      f'{sim_stats["LARGE"]["within_edge_pct"]:.0%}, đúng như tham số within_mask=0.85 đã khai báo.')}
</section>'''

st_rows = [(k, st_stats[k]["n"], f'{st_stats[k]["lost_rate"]:.1%}',
            f'{st_stats[k]["reach_base"]:.1%}', f'{st_stats[k]["reach_graph"]:.1%}') for k in SCALES]
st_channel_bar = grouped_bar_chart(
    [dict(label=k, values=dict(guarantor=st_stats[k]["via_guarantor"] * 100,
                                person=st_stats[k]["via_person"] * 100,
                                address=st_stats[k]["via_address"] * 100)) for k in SCALES],
    [("guarantor", ACCENT2, "Người bảo lãnh"), ("person", ACCENT, "Bất kỳ người liên quan"),
     ("address", GOOD, "Địa chỉ dùng chung")],
    fmt="{:.1f}", unit="%")

section_st = f'''<section>
{section_head("02", "skip_tracing.py — Kênh liên hệ gián tiếp",
  "Khách hàng, số điện thoại, người liên quan/bảo lãnh, địa chỉ dùng chung",
  "Baseline chỉ dùng 1 kênh (địa chỉ); graph hợp 3 kênh độc lập xác suất (bảo lãnh / người liên quan / địa "
  "chỉ dùng chung). Vì 3 kênh độc lập, reach_graph cao hơn baseline gần như chắc chắn về mặt xác suất thuần, "
  "không chỉ nhờ suy luận quan hệ phức tạp.")}
{table(["Quy mô", "n khách hàng", "Tỷ lệ mất liên lạc", "Reach base", "Reach graph"], st_rows)}
<div class="crit-card">
  <div class="crit-name">Đóng góp từng kênh trong nhóm mất liên lạc</div>
  {st_channel_bar}
</div>
{note(f'Kênh "bất kỳ người liên quan" luôn đóng góp cao nhất (~{st_stats["LARGE"]["via_person"]:.0%} ở Large), '
      f'cao hơn cả kênh người bảo lãnh dù người bảo lãnh là kênh "đáng tin" hơn về nghiệp vụ — vì group người '
      f'liên quan có thể gồm 0–2 người, xác suất ít nhất 1 người còn active cộng dồn nhanh. Diễn giải hợp lý: '
      f'kết quả minh hoạ giá trị của việc liên kết nhiều nguồn liên hệ, hơn là bằng chứng về sức mạnh suy luận '
      f'đồ thị nói chung.')}
</section>'''

ir_rows = [(k, ir_stats[k]["n_persons"], ir_stats[k]["n_records"], ir_stats[k]["n_dup_persons"],
            f'{ir_stats[k]["dup_rate"]:.1%}') for k in SCALES]
ir_noise_bar = grouped_bar_chart(
    [dict(label=k, values=dict(phone=ir_stats[k]["same_phone_rate"] * 100,
                                addr=ir_stats[k]["same_addr_rate"] * 100,
                                nid=ir_stats[k]["same_nid_rate"] * 100)) for k in SCALES],
    [("phone", ACCENT, "Giữ cùng SĐT"), ("addr", ACCENT2, "Giữ cùng địa chỉ"),
     ("nid", GOOD, "Giữ cùng số định danh")],
    fmt="{:.1f}", unit="%")
ir_collision_rows = [(k, ir_stats[k]["phone_collision"], ir_stats[k]["addr_collision"]) for k in SCALES]

section_ir = f'''<section>
{section_head("03", "identity_resolution.py — Hợp nhất định danh",
  "Bản ghi trùng lặp có nhiễu thực tế: đổi SĐT, đổi địa chỉ, lỗi gõ số định danh",
  "~14% người có 2 bản ghi. Với mỗi cặp trùng, dữ liệu mô phỏng nhiễu có chủ đích trên từng trường — đây là "
  "phần dữ liệu duy nhất trong repo có nhiễu thực tế được thiết kế riêng, khác hẳn phần dữ liệu tài chính "
  "sạch ở sim_data.py.")}
{table(["Quy mô", "n người thật", "n bản ghi", "n người có bản ghi trùng", "Tỷ lệ trùng"], ir_rows)}
<div class="crit-card">
  <div class="crit-name">Tỷ lệ cặp bản ghi trùng vẫn giữ nguyên giá trị trường (theo trường)</div>
  {ir_noise_bar}
</div>
{table(["Quy mô", "Số điện thoại bị trùng ≥2 người khác nhau", "Địa chỉ bị trùng ≥2 người khác nhau"],
       ir_collision_rows)}
{note(f'Số định danh (national_id) là trường ổn định nhất giữa 2 bản ghi trùng '
      f'({ir_stats["LARGE"]["same_nid_rate"]:.0%} ở Large) — vậy baseline exact-match trên ID vẫn đúng cho '
      f'phần lớn trường hợp, chỉ sai khi rơi vào ~40% ca có lỗi gõ. Ngược lại SĐT/địa chỉ đổi khá thường xuyên '
      f'nên graph_match (nối theo SĐT/địa chỉ) không bắt được toàn bộ recall còn thiếu. Ở quy mô Large đã xuất '
      f'hiện {ir_stats["LARGE"]["phone_collision"]} số điện thoại và {ir_stats["LARGE"]["addr_collision"]} địa '
      f'chỉ bị 2 người khác nhau cùng dùng một cách tình cờ — đây là nguồn gây giảm precision khi n lớn.')}
</section>'''

kpis = "".join([
    kpi("Seed cố định", "42", "toàn bộ số liệu tái lập được"),
    kpi("Quy mô", "800 / 3.200 / 12.800", "SMALL / MEDIUM / LARGE"),
    kpi("Nguồn sinh dữ liệu", "3 module độc lập", "sim_data · skip_tracing · identity_resolution"),
    kpi("Loại dữ liệu", "100% giả lập", "không dùng dữ liệu khách hàng thật"),
])

html = f'''<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8">
<title>Data Context Dashboard — Graph Analytics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{color-scheme:dark;}}
*{{box-sizing:border-box;}}
body{{margin:0;background:{INK};color:{TEXT};font-family:'IBM Plex Sans',sans-serif;}}
.wrap{{max-width:1180px;margin:0 auto;padding:40px 28px 80px;}}
header{{border-bottom:1px solid {LINE};padding-bottom:24px;margin-bottom:28px;}}
h1{{font-size:22px;margin:0 0 4px;font-weight:700;letter-spacing:-.01em;}}
.sub{{color:{MUTED};font-size:13px;font-family:'IBM Plex Mono',monospace;}}
.eyebrow{{color:{ACCENT};font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.14em;
          text-transform:uppercase;margin-bottom:6px;}}
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:28px;}}
.kpi{{background:{PANEL};border:1px solid {LINE};border-radius:10px;padding:16px 18px;}}
.kpi-label{{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:{MUTED};text-transform:uppercase;
           letter-spacing:.08em;margin-bottom:6px;}}
.kpi-value{{font-size:18px;font-weight:700;color:{ACCENT};}}
.kpi-sub{{font-size:11.5px;color:{MUTED};margin-top:4px;}}
.callout{{background:{PANEL};border:1px solid {LINE};border-left:3px solid {ACCENT};border-radius:8px;
         padding:16px 20px;margin-bottom:28px;font-size:13.5px;line-height:1.75;color:#C3CCD6;}}
section{{margin-bottom:52px;}}
.prob-head{{display:flex;gap:16px;align-items:flex-start;margin-bottom:18px;padding-bottom:14px;
           border-bottom:1px solid {LINE};}}
.prob-num{{font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:700;color:{ACCENT2};
          background:{ACCENT2}18;border:1px solid {ACCENT2}44;border-radius:6px;padding:2px 12px;flex-shrink:0;}}
.prob-title{{font-size:19px;font-weight:700;}}
.prob-sub{{font-size:13px;color:{MUTED};margin-top:2px;}}
.prob-desc{{font-size:13px;color:#C3CCD6;line-height:1.7;margin-top:8px;max-width:900px;}}
.crit-card{{background:{PANEL};border:1px solid {LINE};border-radius:10px;padding:20px 22px;margin-bottom:16px;}}
.crit-name{{font-size:14.5px;font-weight:700;margin-bottom:8px;}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;font-family:'IBM Plex Mono',monospace;font-size:11px;color:{MUTED};margin-top:6px;}}
.legend i{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:-1px;}}
table{{width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:11.8px;}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid {LINE};vertical-align:top;}}
th{{color:{MUTED};font-weight:500;text-transform:uppercase;font-size:10px;letter-spacing:.06em;}}
.panel-table{{background:{PANEL};border:1px solid {LINE};border-radius:8px;overflow:hidden;margin-bottom:16px;}}
.note{{background:#0E1116;border:1px solid {LINE};border-left:3px solid {ACCENT};border-radius:8px;
      padding:14px 18px;margin-top:8px;font-size:13px;line-height:1.75;color:#C3CCD6;}}
.note b{{color:#EDF1F5;}}
.note.good{{border-left-color:{GOOD};}}
.note .lbl{{display:block;font-family:'IBM Plex Mono',monospace;font-size:10px;color:{ACCENT};
           text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;}}
.note.good .lbl{{color:{GOOD};}}
footer{{color:{MUTED};font-family:'IBM Plex Mono',monospace;font-size:11px;margin-top:40px;}}
</style></head>
<body><div class="wrap">
<header>
  <div class="eyebrow">Data Context — không phải kết quả mô hình</div>
  <h1>Bối cảnh & phương pháp sinh dữ liệu mô phỏng</h1>
  <div class="sub">Toàn bộ số liệu trong trang này được tính trực tiếp từ generate() của từng module ở mỗi lần chạy</div>
</header>
<div class="kpi-row">{kpis}</div>
<div class="callout">
  Trang này mô tả <b>bản chất và cách sinh ra dữ liệu</b> dùng cho 3 bài toán (Supply Chain/Segmentation, Skip
  Tracing, Identity Resolution) — không phải kết quả AUC/F1/uplift của mô hình (xem dashboard.html cho phần đó).
  Vì dữ liệu là giả lập, các con số % ở đây phản ánh <b>tham số được chọn khi thiết kế mô phỏng</b>, không phải
  fact thị trường thật.
</div>
{section_sim}
{section_st}
{section_ir}
<footer>Generated directly from sim_data.py, skip_tracing.py, identity_resolution.py — synthetic data, seed=42.</footer>
</div></body></html>
'''

with open("dashboard_data.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Saved dashboard_data.html")