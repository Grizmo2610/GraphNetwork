import json

from config import config
INK = "#0B0E11"; PANEL = "#12161C"; LINE = "#232B34"; TEXT = "#D7DEE6"; MUTED = "#7C8A99"
GOOD = "#33D6A6"; BAD = "#F0654A"; ACCENT = "#4FA8FF"; ACCENT2 = "#C792EA"

d = json.load(open(config.all_metric_path))


def tag(ok, t="PASS", f="FAIL"):
    return f'<span class="tag {"pass" if ok else "fail"}">{t if ok else f}</span>'


def pfmt(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.4f}"


def bar_chart(rows, w=740, h=200, unit="", fmt="{:.4f}"):
    pad_l, pad_r, pad_t, pad_b = 24, 20, 30, 30
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    vals = [v for r in rows for v in (r["base"], r["graph"])]
    vmax = max(max(vals), 1e-9) * 1.35
    n = len(rows)
    slot = plot_w / n
    parts = []
    for i, r in enumerate(rows):
        gx = pad_l + i * slot
        bw = slot * 0.28
        for j, (key, color) in enumerate([("base", MUTED), ("graph", ACCENT)]):
            v = r[key]
            bh = max(v, 0) / vmax * plot_h
            bx = gx + slot * 0.5 - bw * 1.1 + j * bw * 1.2
            by = pad_t + plot_h - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" rx="2"/>')
            parts.append(f'<text x="{bx+bw/2:.1f}" y="{by-6:.1f}" text-anchor="middle" font-size="11" '
                         f'font-weight="600" fill="{color}" font-family="IBM Plex Mono,monospace">{fmt.format(v)}{unit}</text>')
        parts.append(f'<text x="{gx+slot/2:.1f}" y="{h-10}" text-anchor="middle" font-size="12" '
                     f'fill="{TEXT}" font-family="IBM Plex Mono,monospace" font-weight="600">{r["label"]}</text>')
    base_line = pad_t + plot_h
    parts.append(f'<line x1="{pad_l}" x2="{w-pad_r}" y1="{base_line}" y2="{base_line}" stroke="{LINE}" stroke-width="1"/>')
    return f'''<svg viewBox="0 0 {w} {h}" width="100%">{''.join(parts)}</svg>
      <div class="legend"><span><i style="background:{MUTED}"></i>Baseline</span><span><i style="background:{ACCENT}"></i>+ Graph</span></div>'''


def ci_chart(rows, w=740, h=190):
    pad_l, pad_r, pad_t, pad_b = 20, 20, 20, 30
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b
    los = [r["ci"][0] for r in rows]; his = [r["ci"][1] for r in rows]
    span = max(max(his), 0) - min(min(los), 0) or 1
    vmin, vmax = min(min(los), 0) - span * 0.25, max(max(his), 0) + span * 0.25

    def yx(v):
        return pad_t + plot_h - (v - vmin) / (vmax - vmin) * plot_h

    zero_y = yx(0)
    n = len(rows)
    slot = plot_w / n
    parts = [f'<line x1="{pad_l}" x2="{w-pad_r}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" stroke="{MUTED}" '
             f'stroke-width="1" stroke-dasharray="4,4"/>',
             f'<text x="{pad_l-4}" y="{zero_y+4:.1f}" text-anchor="end" font-size="10" fill="{MUTED}" '
             f'font-family="IBM Plex Mono,monospace">0</text>']
    for i, r in enumerate(rows):
        cx = pad_l + slot * (i + 0.5)
        y_lo, y_hi, y_val = yx(r["ci"][0]), yx(r["ci"][1]), yx(r["val"])
        ok = r["ci"][0] > 0
        c = GOOD if ok else BAD
        parts.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y_hi:.1f}" y2="{y_lo:.1f}" stroke="{c}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{cx:.1f}" cy="{y_val:.1f}" r="5" fill="{c}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y_hi-8:.1f}" text-anchor="middle" font-size="10.5" font-weight="600" '
                     f'fill="{c}" font-family="IBM Plex Mono,monospace">{r["ci"][1]:+.4f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{y_lo+16:.1f}" text-anchor="middle" font-size="10.5" font-weight="600" '
                     f'fill="{c}" font-family="IBM Plex Mono,monospace">{r["ci"][0]:+.4f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{h-8}" text-anchor="middle" font-size="12" fill="{TEXT}" '
                     f'font-family="IBM Plex Mono,monospace" font-weight="600">{r["label"]} &nbsp; uplift {r["val"]:+.4f}</text>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%">{"".join(parts)}</svg>'


def problem_header(num, title, subtitle, desc):
    return f'''
    <div class="prob-head">
      <span class="prob-num">{num}</span>
      <div><div class="prob-title">{title}</div><div class="prob-sub">{subtitle}</div>
      <div class="prob-desc">{desc}</div></div>
    </div>'''


# ---------- 1. Supply Chain Finance ----------
sc = {s["name"]: s for s in d["supply_chain"]["scales"]}
sc_rows_auc = "".join(
    f'<tr><td>{k}</td><td>{sc[k]["auc_base"]:.4f}</td><td>{sc[k]["auc_graph"]:.4f}</td>'
    f'<td class="{"ok" if sc[k]["auc_uplift"]>0 else "no"}">{sc[k]["auc_uplift"]:+.4f}</td>'
    f'<td>{pfmt(sc[k]["delong_p"])}</td><td>{tag(sc[k]["pass_auc"])}</td></tr>' for k in ["SMALL", "MEDIUM", "LARGE"])
sc_auc_bar = bar_chart([dict(label=k, base=sc[k]["auc_base"], graph=sc[k]["auc_graph"]) for k in ["SMALL", "MEDIUM", "LARGE"]])
sc_f1_ci = ci_chart([dict(label=k, val=sc[k]["f1_uplift"], ci=sc[k]["f1_ci"]) for k in ["SMALL", "MEDIUM", "LARGE"]])
sc_ks_ci = ci_chart([dict(label=k, val=sc[k]["ks_uplift"], ci=sc[k]["ks_ci"]) for k in ["SMALL", "MEDIUM", "LARGE"]])
sc_lift_ci = ci_chart([dict(label=k, val=sc[k]["lift_uplift"], ci=sc[k]["lift_ci"]) for k in ["SMALL", "MEDIUM", "LARGE"]])
sc_summary_rows = "".join(
    f'<tr><td>{k}</td><td>{tag(sc[k]["pass_auc"])}</td><td>{tag(sc[k]["pass_f1"])}</td>'
    f'<td>{tag(sc[k]["pass_ks"])}</td><td>{tag(sc[k]["pass_lift"])}</td><td>{tag(sc[k]["graph_in_top5"])}</td>'
    f'<td>{tag(sc[k]["accept"], "ACCEPT", "REJECT")}</td></tr>' for k in ["SMALL", "MEDIUM", "LARGE"])

section_supply_chain = f'''
<section>
  {problem_header("01", "Supply Chain Finance", "Dự đoán rủi ro default của SME dựa trên vị trí trong chuỗi cung ứng",
    "Bài toán phân loại nhị phân (default có/không), nên áp dụng nguyên khung 5 tiêu chí "
    "AUC-ROC uplift / F1 / KS / Lift@10% / SHAP-rank như một mô hình chấm điểm tín dụng chuẩn — chỉ khác ở chỗ "
    "thực thể là <b>doanh nghiệp SME</b> và cạnh đồ thị là quan hệ <b>mua–bán trong chuỗi cung ứng</b>, "
    "đặc trưng graph có thêm <b>concentration</b> (tỷ trọng giao dịch dồn vào một đối tác duy nhất — đo mức phụ thuộc)."
  )}
  <div class="crit-card">
    <div class="crit-name">AUC-ROC Uplift</div>
    <div class="crit-formula">AUC<sub>graph</sub> − AUC<sub>baseline</sub> &gt; 0, DeLong test <b>p &lt; 0,05</b></div>
    <div class="panel-table"><table>
      <tr><th>Quy mô</th><th>AUC base</th><th>AUC graph</th><th>Uplift</th><th>DeLong p</th><th>Kết quả</th></tr>
      {sc_rows_auc}
    </table></div>
    <div class="chart-title">Baseline vs Graph</div>{sc_auc_bar}
  </div>
  <div class="crit-card">
    <div class="crit-name">F1 / KS / Lift@10% Uplift — Bootstrap CI 95%</div>
    <div class="chart-title">F1-score uplift</div>{sc_f1_ci}
    <div class="chart-title">KS Statistic uplift</div>{sc_ks_ci}
    <div class="chart-title">Lift@10% uplift</div>{sc_lift_ci}
  </div>
  <div class="panel-table"><table>
    <tr><th>Quy mô</th><th>AUC</th><th>F1</th><th>KS</th><th>Lift@10</th><th>Top-5 SHAP</th><th>Quyết định</th></tr>
    {sc_summary_rows}
  </table></div>
  <div class="note {"good" if sc["LARGE"]["accept"] else "warn"}">
    <span class="lbl">Đọc số liệu</span>
    Ở Small, AUC tăng +{sc["SMALL"]["auc_uplift"]:.4f} nhưng DeLong p={pfmt(sc["SMALL"]["delong_p"])} chưa dưới 0,05.
    concentration (mức phụ thuộc vào một đối tác duy nhất) cùng detected_community luôn nằm trong top-5 SHAP ở cả
    ba quy mô — đúng với giả thuyết "SME phụ thuộc một khách hàng lớn mang rủi ro khác hẳn SME có danh mục đa
    dạng" dù hồ sơ tài chính giống nhau. Chỉ tới Large, cả 5 tiêu chí mới đạt đồng thời
    (DeLong p={pfmt(sc["LARGE"]["delong_p"])}) → <b>{"ACCEPT" if sc["LARGE"]["accept"] else "REJECT"}</b>.
  </div>
</section>'''

# ---------- 2. Skip Tracing ----------
st = {s["name"]: s for s in d["skip_tracing"]["scales"]}
st_rows = "".join(
    f'<tr><td>{k}</td><td>{st[k]["n_lost_contact"]:,} / {st[k]["n"]:,} ({st[k]["lost_contact_rate"]:.1%})</td>'
    f'<td>{st[k]["reach_base"]:.4f}</td><td>{st[k]["reach_graph"]:.4f}</td>'
    f'<td class="ok">{st[k]["uplift"]:+.4f}</td><td>[{st[k]["ci"][0]:+.4f}, {st[k]["ci"][1]:+.4f}]</td>'
    f'<td>{st[k]["residual_manual_review"]:.1%}</td><td>{tag(st[k]["pass_uplift"])}</td></tr>'
    for k in ["SMALL", "MEDIUM", "LARGE"])
st_bar = bar_chart([dict(label=k, base=st[k]["reach_base"], graph=st[k]["reach_graph"]) for k in ["SMALL", "MEDIUM", "LARGE"]])
st_ci = ci_chart([dict(label=k, val=st[k]["uplift"], ci=st[k]["ci"]) for k in ["SMALL", "MEDIUM", "LARGE"]])
chan_rows = "".join(
    f'<tr><td>{k}</td><td>{st[k]["channel_breakdown"]["guarantor_phone"]:.1%}</td>'
    f'<td>{st[k]["channel_breakdown"]["any_person_phone"]:.1%}</td>'
    f'<td>{st[k]["channel_breakdown"]["shared_address"]:.1%}</td></tr>' for k in ["SMALL", "MEDIUM", "LARGE"])

section_skip_tracing = f'''
<section>
  {problem_header("02", "Skip Tracing", "Tìm kênh liên hệ gián tiếp cho khách hàng nợ xấu đã mất liên lạc",
    "Không phải bài toán phân loại — là bài toán <b>khả năng tiếp cận</b> (reachability). Baseline chỉ dùng "
    "kênh liên hệ trực tiếp (SĐT/địa chỉ đăng ký của chính khách hàng); Graph truy ngược qua "
    "<b>người bảo lãnh, người thân, hoặc người khác chung địa chỉ</b> để tìm SĐT còn hoạt động.")}
  <div class="crit-card">
    <div class="crit-name">Reach-rate Uplift (trên nhóm khách hàng đã mất liên lạc)</div>
    <div class="crit-formula">Reach<sub>graph</sub> − Reach<sub>baseline</sub>, tính trên tập khách hàng
      lost-contact (toàn bộ SĐT chính chủ đều inactive); bootstrap CI 95% không chứa 0</div>
    <div class="panel-table"><table>
      <tr><th>Quy mô</th><th>Số hồ sơ mất liên lạc</th><th>Reach base</th><th>Reach graph</th><th>Uplift</th>
        <th>95% CI</th><th>Còn lại cần kiểm tra tay</th><th>Kết quả</th></tr>
      {st_rows}
    </table></div>
    <div class="chart-title">Reach-rate: Baseline vs Graph</div>{st_bar}
    <div class="chart-title">Uplift &amp; khoảng tin cậy Bootstrap 95%</div>{st_ci}
  </div>
  <div class="panel-table"><table>
    <tr><th>Quy mô</th><th>Qua SĐT người bảo lãnh</th><th>Qua SĐT bất kỳ người liên quan</th><th>Qua địa chỉ dùng chung</th></tr>
    {chan_rows}
  </table></div>
  <div class="note good">
    <span class="lbl">Đọc số liệu</span>
    Đây là bài toán duy nhất trong 4 bài toán <b>đạt tiêu chí ở cả ba quy mô</b> — khác hẳn Supply Chain Finance
    hay Segmentation. Lý do: hiệu ứng "tìm được thêm 1 kênh liên hệ" không phụ thuộc vào việc mạng lưới có đủ
    lớn để lộ ra cấu trúc thống kê hay không — chỉ cần khách hàng <i>có ít nhất một</i> người bảo lãnh/người thân
    còn liên lạc được là đã tạo ra khác biệt ngay lập tức. Ở Large, reach-rate tăng từ
    {st["LARGE"]["reach_base"]:.1%} lên {st["LARGE"]["reach_graph"]:.1%}, tỷ lệ hồ sơ phải đi xác minh thủ công
    giảm còn {st["LARGE"]["residual_manual_review"]:.1%}.
  </div>
</section>'''

# ---------- 3. Customer Segmentation ----------
sg = {s["name"]: s for s in d["segmentation"]["scales"]}
sg_rows = "".join(
    f'<tr><td>{k}</td><td>{sg[k]["ari_base"]:.4f}</td><td>{sg[k]["ari_graph"]:.4f}</td>'
    f'<td class="{"ok" if sg[k]["ari_uplift"]>0 else "no"}">{sg[k]["ari_uplift"]:+.4f}</td>'
    f'<td>[{sg[k]["ari_ci"][0]:+.4f}, {sg[k]["ari_ci"][1]:+.4f}]</td><td>{tag(sg[k]["pass_ari"])}</td></tr>'
    for k in ["SMALL", "MEDIUM", "LARGE"])
sg_ari_bar = bar_chart([dict(label=k, base=sg[k]["ari_base"], graph=sg[k]["ari_graph"]) for k in ["SMALL", "MEDIUM", "LARGE"]], fmt="{:.4f}")
sg_sil_bar = bar_chart([dict(label=k, base=sg[k]["sil_base"], graph=sg[k]["sil_graph"]) for k in ["SMALL", "MEDIUM", "LARGE"]], fmt="{:.4f}")
sg_adopt_rows = "".join(
    f'<tr><td>{k}</td><td>{sg[k]["adopt_auc_base"]:.4f}</td><td>{sg[k]["adopt_auc_graph"]:.4f}</td>'
    f'<td class="ok">{sg[k]["adopt_auc_uplift"]:+.4f}</td><td>{pfmt(sg[k]["adopt_delong_p"])}</td>'
    f'<td>{tag(sg[k]["pass_adopt"])}</td></tr>' for k in ["SMALL", "MEDIUM", "LARGE"])
sg_adopt_bar = bar_chart([dict(label=k, base=sg[k]["adopt_auc_base"], graph=sg[k]["adopt_auc_graph"]) for k in ["SMALL", "MEDIUM", "LARGE"]])

section_segmentation = f'''
<section>
  {problem_header("03", "Customer Segmentation", "Phân khúc khách hàng theo vị trí mạng lưới thay vì chỉ nhân khẩu học",
    "Không có nhãn đúng/sai — đây là bài toán <b>phân cụm không giám sát</b>, nên dùng Adjusted Rand Index (ARI) "
    "so với phân khúc ẩn thực sự (ground-truth, không lộ ra như một feature) và Silhouette score. Kèm theo một "
    "bài toán phụ mang tính ứng dụng: <b>dự đoán khả năng mua sản phẩm tiếp theo</b> dựa trên hành vi lân cận "
    "(neighbor adoption rate) — đúng ý &quot;nếu phần lớn người trong mạng lưới vừa mở tài khoản tiết kiệm, xác "
    "suất khách hàng đó có nhu cầu tương tự cao hơn mức nền&quot;.")}
  <div class="crit-card">
    <div class="crit-name">Adjusted Rand Index (ARI) — chất lượng phân cụm so với phân khúc ẩn thực sự</div>
    <div class="crit-formula">ARI<sub>graph</sub> − ARI<sub>baseline</sub>, bootstrap CI 95% không chứa 0</div>
    <div class="panel-table"><table>
      <tr><th>Quy mô</th><th>ARI (demo-only)</th><th>ARI (+graph)</th><th>Uplift</th><th>95% CI</th><th>Kết quả</th></tr>
      {sg_rows}
    </table></div>
    <div class="chart-title">ARI: demo-only vs +graph feature</div>{sg_ari_bar}
    <div class="chart-title">Silhouette score (chất lượng phân cụm nội tại, không cần ground-truth)</div>{sg_sil_bar}
  </div>
  <div class="crit-card">
    <div class="crit-name">Bài toán phụ: dự đoán mua sản phẩm tiếp theo (AUC uplift)</div>
    <div class="crit-formula">AUC<sub>graph</sub> − AUC<sub>baseline</sub> &gt; 0, DeLong test p &lt; 0,05 —
      đặc trưng graph thêm vào: neighbor_adopt_rate (tỷ lệ hàng xóm giao dịch đã mua sản phẩm), degree, pagerank</div>
    <div class="panel-table"><table>
      <tr><th>Quy mô</th><th>AUC base</th><th>AUC graph</th><th>Uplift</th><th>DeLong p</th><th>Kết quả</th></tr>
      {sg_adopt_rows}
    </table></div>
    <div class="chart-title">Baseline vs Graph</div>{sg_adopt_bar}
  </div>
  <div class="note">
    <span class="lbl">Đọc số liệu</span>
    ARI cả hai phương pháp đều <b>rất thấp</b> (gần 0) — đây thực ra là điều đúng như kỳ vọng: phân khúc ẩn trong
    mô phỏng này (giống các "nhóm giao dịch với nhau" trong thực tế) <b>không có điểm chung nhân khẩu học rõ
    ràng</b>, nên phân cụm chỉ dựa vào debt/income/age gần như ngẫu nhiên (ARI base ≈ 0). Thêm đặc trưng đồ thị
    cải thiện ARI một cách khiêm tốn nhưng có ý nghĩa thống kê ở Large (CI [{sg["LARGE"]["ari_ci"][0]:+.4f},
    {sg["LARGE"]["ari_ci"][1]:+.4f}]). Silhouette score lại <b>giảm</b> khi thêm graph feature — dễ hiểu vì
    silhouette đo độ "gọn" của cụm trong không gian đặc trưng, còn ARI đo độ khớp với nhóm thật; hai chỉ số này
    không nhất thiết cùng chiều. Ở bài toán phụ (dự đoán mua sản phẩm), tín hiệu graph rõ ràng hơn nhiều: AUC
    tăng đều, đạt ý nghĩa thống kê từ Medium (p={pfmt(sg["MEDIUM"]["adopt_delong_p"])}).
  </div>
</section>'''

# ---------- 4. Customer Relationship ----------
ir = {s["name"]: s for s in d["identity_resolution"]["scales"]}
ir_rows = "".join(
    f'<tr><td>{k}</td><td>{ir[k]["n_records"]:,} bản ghi / {ir[k]["n_true_persons"]:,} người thật '
    f'({ir[k]["n_duplicated_persons"]:,} bị trùng)</td>'
    f'<td>{ir[k]["precision_base"]:.4f} → {ir[k]["precision_graph"]:.4f}</td>'
    f'<td>{ir[k]["recall_base"]:.4f} → {ir[k]["recall_graph"]:.4f}</td>'
    f'<td>{ir[k]["f1_base"]:.4f} → {ir[k]["f1_graph"]:.4f}</td>'
    f'<td class="ok">{ir[k]["f1_uplift"]:+.4f}</td>'
    f'<td>[{ir[k]["f1_ci"][0]:+.4f}, {ir[k]["f1_ci"][1]:+.4f}]</td><td>{tag(ir[k]["pass_f1"])}</td></tr>'
    for k in ["SMALL", "MEDIUM", "LARGE"])
ir_bar = bar_chart([dict(label=k, base=ir[k]["f1_base"], graph=ir[k]["f1_graph"]) for k in ["SMALL", "MEDIUM", "LARGE"]])
ir_ci = ci_chart([dict(label=k, val=ir[k]["f1_uplift"], ci=ir[k]["f1_ci"]) for k in ["SMALL", "MEDIUM", "LARGE"]])

section_identity = f'''
<section>
  {problem_header("04", "Customer Relationship", "Hợp nhất hồ sơ định danh phân mảnh (identity resolution)",
    "Bài toán so khớp thực thể (entity resolution): mỗi &quot;người thật&quot; có thể có nhiều bản ghi khách "
    "hàng trùng lặp (do đăng ký lại, sai lệch CMND/CCCD…). Baseline chỉ gộp theo <b>trùng khớp tuyệt đối số "
    "định danh (national ID)</b>; Graph gộp theo <b>liên kết chuyển tiếp</b> (transitive closure) qua SĐT hoặc "
    "địa chỉ dùng chung. Đánh giá bằng precision/recall/F1 theo cặp bản ghi (pairwise), giống Experian đo tỷ lệ "
    "khớp danh tính.")}
  <div class="crit-card">
    <div class="crit-name">Pairwise Precision / Recall / F1 uplift</div>
    <div class="crit-formula">F1<sub>graph</sub> − F1<sub>baseline</sub>, bootstrap CI 95% không chứa 0</div>
    <div class="panel-table"><table>
      <tr><th>Quy mô</th><th>Quy mô dữ liệu</th><th>Precision (base→graph)</th><th>Recall (base→graph)</th>
        <th>F1 (base→graph)</th><th>F1 uplift</th><th>95% CI</th><th>Kết quả</th></tr>
      {ir_rows}
    </table></div>
    <div class="chart-title">F1: Baseline (exact-match) vs Graph (connected components)</div>{ir_bar}
    <div class="chart-title">Uplift &amp; khoảng tin cậy Bootstrap 95%</div>{ir_ci}
  </div>
  <div class="note {"good" if ir["LARGE"]["pass_f1"] else "warn"}">
    <span class="lbl">Đọc số liệu</span>
    Baseline (khớp national_id tuyệt đối) luôn giữ <b>precision gần như tuyệt đối</b> (100% ở Small/Medium,
    {ir["LARGE"]["precision_base"]*100:.2f}% ở Large) — không bao giờ gộp nhầm, nhưng recall rất thấp
    (chỉ {ir["SMALL"]["recall_base"]:.1%}–{ir["LARGE"]["recall_base"]:.1%}) vì bỏ sót toàn bộ trường hợp có sai
    lệch nhỏ trên số định danh. Graph (liên kết qua SĐT/địa chỉ dùng chung) kéo recall lên
    {ir["LARGE"]["recall_graph"]:.1%} ở Large, đánh đổi một chút precision
    ({ir["LARGE"]["precision_graph"]*100:.2f}%) do một số liên kết chuyển tiếp gộp nhầm hai người khác nhau tình
    cờ dùng chung địa chỉ. F1 tổng hợp chỉ đạt ý nghĩa thống kê ở Large — giống Experian: hạ tầng identity graph
    chỉ thực sự đáng giá khi khối lượng bản ghi đủ lớn để lợi ích recall vượt chi phí precision.
  </div>
</section>'''

# ---------- Cross-problem summary ----------
def decision(problem_scales, key="accept"):
    return {k: problem_scales[k].get(key, problem_scales[k].get("pass_uplift", problem_scales[k].get("pass_f1")))
            for k in ["SMALL", "MEDIUM", "LARGE"]}


sc_dec = {k: sc[k]["accept"] for k in sc}
st_dec = {k: st[k]["pass_uplift"] for k in st}
sg_dec = {k: (sg[k]["pass_ari"] and sg[k]["pass_adopt"]) for k in sg}
ir_dec = {k: ir[k]["pass_f1"] for k in ir}

summary_rows = ""
for k in ["SMALL", "MEDIUM", "LARGE"]:
    summary_rows += (f'<tr><td>{k}</td><td>{tag(sc_dec[k], "ACCEPT", "REJECT")}</td>'
                     f'<td>{tag(st_dec[k], "ACCEPT", "REJECT")}</td>'
                     f'<td>{tag(sg_dec[k], "ACCEPT", "REJECT")}</td>'
                     f'<td>{tag(ir_dec[k], "ACCEPT", "REJECT")}</td></tr>')

html = f'''<!DOCTYPE html>
<html lang="vi"><head><meta charset="UTF-8">
<title>Graph Analytics — 4 Business Problems Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{{color-scheme:dark;}}
  *{{box-sizing:border-box;}}
  body{{margin:0;background:{INK};color:{TEXT};font-family:'IBM Plex Sans',sans-serif;}}
  .wrap{{max-width:1180px;margin:0 auto;padding:40px 28px 80px;}}
  header{{border-bottom:1px solid {LINE};padding-bottom:24px;margin-bottom:32px;}}
  h1{{font-size:22px;margin:0 0 4px;font-weight:700;letter-spacing:-.01em;}}
  .sub{{color:{MUTED};font-size:13px;font-family:'IBM Plex Mono',monospace;}}
  .eyebrow{{color:{ACCENT};font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.14em;
            text-transform:uppercase;margin-bottom:6px;}}
  section{{margin-bottom:52px;}}
  .callout{{background:{PANEL};border:1px solid {LINE};border-left:3px solid {ACCENT};border-radius:8px;
           padding:16px 20px;margin-bottom:28px;font-size:13.5px;line-height:1.75;color:#C3CCD6;}}
  .prob-head{{display:flex;gap:16px;align-items:flex-start;margin-bottom:18px;padding-bottom:14px;
             border-bottom:1px solid {LINE};}}
  .prob-num{{font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:700;color:{ACCENT2};
            background:{ACCENT2}18;border:1px solid {ACCENT2}44;border-radius:6px;padding:2px 12px;flex-shrink:0;}}
  .prob-title{{font-size:19px;font-weight:700;}}
  .prob-sub{{font-size:13px;color:{MUTED};margin-top:2px;}}
  .prob-desc{{font-size:13px;color:#C3CCD6;line-height:1.7;margin-top:8px;max-width:900px;}}
  .crit-card{{background:{PANEL};border:1px solid {LINE};border-radius:10px;padding:20px 22px;margin-bottom:16px;}}
  .crit-name{{font-size:14.5px;font-weight:700;margin-bottom:8px;}}
  .crit-formula{{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:#C3CCD6;background:#0E1116;
                border:1px solid {LINE};border-radius:6px;padding:9px 13px;margin-bottom:12px;line-height:1.6;}}
  .crit-formula sub{{font-size:9px;}}
  .chart-title{{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:{MUTED};margin:14px 0 8px;}}
  .legend{{display:flex;gap:18px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:{MUTED};margin-top:4px;}}
  .legend i{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:-1px;}}
  table{{width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:11.8px;}}
  th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid {LINE};vertical-align:top;}}
  th{{color:{MUTED};font-weight:500;text-transform:uppercase;font-size:10px;letter-spacing:.06em;}}
  td.ok{{color:{GOOD};}} td.no{{color:{BAD};}}
  .panel-table{{background:{PANEL};border:1px solid {LINE};border-radius:8px;overflow:hidden;margin-bottom:16px;}}
  .tag{{display:inline-block;padding:1px 8px;border-radius:3px;font:600 10.5px/1.6 'IBM Plex Mono',monospace;letter-spacing:.03em;}}
  .tag.pass{{background:{GOOD}22;color:{GOOD};border:1px solid {GOOD}55;}}
  .tag.fail{{background:{BAD}22;color:{BAD};border:1px solid {BAD}55;}}
  .note{{background:#0E1116;border:1px solid {LINE};border-left:3px solid {ACCENT};border-radius:8px;
        padding:14px 18px;margin-top:8px;font-size:13px;line-height:1.75;color:#C3CCD6;}}
  .note b{{color:#EDF1F5;}}
  .note.good{{border-left-color:{GOOD};}}
  .note.warn{{border-left-color:{BAD};}}
  .note .lbl{{display:block;font-family:'IBM Plex Mono',monospace;font-size:10px;color:{ACCENT};
             text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;}}
  .note.good .lbl{{color:{GOOD};}}
  .note.warn .lbl{{color:{BAD};}}
  .summary-block{{background:{PANEL};border:1px solid {LINE};border-radius:8px;padding:24px 26px;}}
  .summary-block p{{font-size:14px;line-height:1.8;color:#C3CCD6;margin:0 0 14px;}}
  .summary-block p:last-child{{margin-bottom:0;}}
  .summary-block b{{color:#EDF1F5;}}
  footer{{color:{MUTED};font-family:'IBM Plex Mono',monospace;font-size:11px;margin-top:40px;}}
</style></head>
<body><div class="wrap">
  <header>
    <div class="eyebrow">Graph Analytics — Đánh giá 4 bài toán nghiệp vụ</div>
    <h1>Skip Tracing · Supply Chain Finance · Customer Segmentation · Customer Relationship</h1>
    <div class="sub">Mỗi bài toán dùng đúng bộ chỉ số phù hợp với bản chất của nó — không ép chung một khung AUC/F1/KS/Lift</div>
  </header>

  <div class="callout">
    <b>Vì sao 4 bộ chỉ số khác nhau?</b> Chỉ Supply Chain Finance là bài toán phân loại có nhãn (default/không) nên
    dùng AUC-ROC/F1/KS/Lift@10%/SHAP-rank. Skip Tracing là bài toán khả năng tiếp cận (reachability) → đo bằng
    reach-rate. Customer Segmentation là phân cụm không giám sát → đo bằng Adjusted Rand Index + Silhouette (kèm
    một bài toán phụ có giám sát để minh hoạ ứng dụng). Customer Relationship là so khớp thực thể → đo bằng
    precision/recall/F1 theo cặp bản ghi. Toàn bộ số liệu lấy trực tiếp từ <code>metrics_all.json</code>.
  </div>

  {section_supply_chain}
  {section_skip_tracing}
  {section_segmentation}
  {section_identity}

  <section>
    <div class="prob-head"><span class="prob-num">05</span><div>
      <div class="prob-title">Tổng hợp 4 bài toán theo quy mô</div>
      <div class="prob-sub">Quyết định ACCEPT/REJECT của từng bài toán tại mỗi quy mô dữ liệu</div></div></div>
    <div class="panel-table"><table>
      <tr><th>Quy mô</th><th>Supply Chain Finance</th><th>Skip Tracing</th><th>Customer Segmentation</th><th>Customer Relationship</th></tr>
      {summary_rows}
    </table></div>
    <div class="summary-block">
      <p><b>Skip Tracing là bài toán duy nhất đạt ở cả ba quy mô</b> — giá trị của nó không phụ thuộc vào việc
        mạng đủ lớn để lộ cấu trúc thống kê, chỉ cần tồn tại ít nhất một liên kết người bảo lãnh/người thân còn
        hoạt động. Ba bài toán còn lại (Supply Chain Finance, Segmentation, Customer Relationship) đều theo cùng
        một quy luật: <b>chỉ đạt đầy đủ tiêu chí ở quy mô Large</b>, vì giá trị của chúng đến từ tín hiệu thống
        kê phải đủ mạnh để tách khỏi nhiễu ngẫu nhiên — càng nhiều dữ liệu, tín hiệu cấu trúc mạng lưới càng rõ.</p>
      <p>Điều này gợi ý một nguyên tắc chung khi quyết định đầu tư Graph Analytics: <b>bài toán dạng "tìm kiếm/
        truy vết qua liên kết đã biết" (Skip Tracing, và ở mức độ nào đó Customer Relationship) tạo giá trị gần
        như ngay lập tức</b>; còn <b>bài toán dạng "phát hiện tín hiệu ẩn trong cấu trúc mạng" (Supply Chain
        Finance, Segmentation) cần một khối lượng dữ liệu đủ lớn mới chứng minh được hiệu quả có ý nghĩa thống
        kê</b>, dù xu hướng cải thiện đã xuất hiện từ sớm.</p>
    </div>
  </section>

  <footer>Generated from metrics_all.json — synthetic data, offline demo pipeline (4 modules: supply_chain.py, skip_tracing.py, segmentation.py, identity_resolution.py).</footer>
</div></body></html>
'''

with open("output/dashboard.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Saved dashboard.html")