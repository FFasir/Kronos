/**
 * PctText — 涨跌幅文本（A 股约定：红涨绿跌）
 *
 * 统一全站涨跌颜色口径，替代散落各处的 var(--success)/var(--danger) 三元表达式。
 *
 * 用法:
 *   <PctText value={0.0325} />                    // +3.25%（红）
 *   <PctText value={-0.012} bold />               // -1.20%（绿，加粗）
 *   <PctText value={5.2} raw suffix="%" />        // 已是百分数时传 raw
 */

export interface PctTextProps {
  /** 数值。默认视为小数收益率（0.03 → 3.00%）；raw=true 时按原值显示 */
  value: number | null | undefined
  /** value 已经是百分数（不再 ×100） */
  raw?: boolean
  /** 小数位，默认 2 */
  precision?: number
  /** 加粗 */
  bold?: boolean
  /** 显示 + 号，默认 true */
  showSign?: boolean
  /** 后缀，默认 % */
  suffix?: string
}

/** A 股红涨绿跌 */
export function pctColor(value: number): string {
  return value > 0 ? 'var(--danger)' : value < 0 ? 'var(--success)' : 'var(--text-muted)'
}

export default function PctText({
  value,
  raw = false,
  precision = 2,
  bold = false,
  showSign = true,
  suffix = '%',
}: PctTextProps) {
  if (value == null || Number.isNaN(value)) {
    return <span style={{ color: 'var(--text-muted)' }}>--</span>
  }
  const pct = raw ? value : value * 100
  const sign = showSign && pct > 0 ? '+' : ''
  return (
    <span style={{ color: pctColor(pct), fontWeight: bold ? 600 : undefined }}>
      {sign}
      {pct.toFixed(precision)}
      {suffix}
    </span>
  )
}
