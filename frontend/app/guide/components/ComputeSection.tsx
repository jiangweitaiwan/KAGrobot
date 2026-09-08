/**
 * ComputeSection.tsx — Section 4:「可計算性」。
 *
 * 回收「送出 RC skill」這個動作的意義:不是又進入一輪對話,而是
 * 套用一個具體的量化模型算出數字。目前只有第一列（升息衝擊分析／
 * Dornbusch 匯率超貶模型）是目錄裡確認存在的真實 skill,其餘兩列
 * 先用〔待補〕佔位,避免誤導使用者以為是已上線的功能——待補上
 * 真實的 RC skill 名稱與對應模型後,直接取代 COMPARISON_ROWS 裡
 * 標記 placeholder 的兩筆即可。
 */

type ComparisonRow = { rc: string; model: string; output: string; placeholder?: boolean };

const COMPARISON_ROWS: ComparisonRow[] = [
  {
    rc: "升息衝擊分析",
    model: "Dornbusch 匯率超貶模型",
    output: "短期超貶幅度、收斂速度",
  },
  { rc: "〔待補：RC skill 名稱〕", model: "〔待補：對應模型〕", output: "〔待補：輸出型態〕", placeholder: true },
  { rc: "〔待補：RC skill 名稱〕", model: "〔待補：對應模型〕", output: "〔待補：輸出型態〕", placeholder: true },
];

export default function ComputeSection() {
  return (
    <div className="guide-compute">
      <div className="guide-compute-result">
        <div className="guide-compute-result-left">
          <span className="guide-compute-tag">RC Skill</span>
          <p className="guide-compute-rc">升息衝擊分析</p>
          <span className="guide-compute-tag">套用模型</span>
          <p className="guide-compute-model">Dornbusch 匯率超貶模型</p>
        </div>
        <div className="guide-compute-result-right">
          <span className="guide-compute-tag">計算結論（示意）</span>
          <p className="guide-compute-output">
            升息 100bp → 匯率短期超貶約 <strong>X%</strong>，收斂需時 <strong>Y</strong> 季
          </p>
          <div className="guide-compute-chart" aria-hidden="true">
            <svg viewBox="0 0 220 90">
              <polyline
                points="0,20 40,10 70,60 110,70 150,55 190,45 220,42"
                fill="none"
                stroke="#ffcc4d"
                strokeWidth="2"
              />
              <line x1="0" y1="45" x2="220" y2="45" stroke="rgba(232,238,248,0.2)" strokeDasharray="2,3" />
            </svg>
          </div>
        </div>
      </div>

      <p className="guide-note">
        這就是第一區塊步驟⑥「送出」之後，系統實際在做的事——每個 RC 背後都有一個真正會跑數字的模型，不是空泛建議。
      </p>

      <table className="guide-compute-table">
        <thead>
          <tr>
            <th>RC Skill</th>
            <th>背後模型</th>
            <th>輸出</th>
          </tr>
        </thead>
        <tbody>
          {COMPARISON_ROWS.map((row, i) => (
            <tr key={i} className={row.placeholder ? "guide-row-placeholder" : undefined}>
              <td>{row.rc}</td>
              <td>{row.model}</td>
              <td>{row.output}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
