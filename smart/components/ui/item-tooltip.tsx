"use client"

import React from "react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ItemScores {
  reliability: number
  maintainability: number
  availability: number
  vulnerability: number
  overall: number
}

export interface ItemCriticalityLevels {
  reliability: string
  maintainability: string
  availability: string
  vulnerability: string
  overall: string
}

export interface ItemTooltipData {
  /** The component type: Application, Node, Topic, Broker, Library */
  type: string
  /** Display name (falls back to id if absent) */
  name?: string
  /** Raw RMAV quality scores [0–1], where lower = better (risk scale) */
  scores?: ItemScores
  /** Per-dimension criticality labels */
  criticality_levels?: ItemCriticalityLevels
  /** Overall criticality label (used when criticality_levels is absent) */
  criticality_level?: string
  /** Additional structural / type-specific numeric metrics */
  metrics?: Record<string, number | boolean | string | null | undefined>
  /** Raw node properties (type-specific attributes) */
  properties?: Record<string, unknown>
}

export interface ItemTooltipProps {
  data: ItemTooltipData
  children: React.ReactNode
  side?: "top" | "right" | "bottom" | "left"
  className?: string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const CRITICALITY_COLOR: Record<string, string> = {
  critical: "bg-red-500/15 text-red-500",
  high:     "bg-orange-500/15 text-orange-500",
  medium:   "bg-amber-500/15 text-amber-500",
  low:      "bg-emerald-500/15 text-emerald-500",
  minimal:  "bg-zinc-500/15 text-zinc-400",
}

function critColor(level: string | undefined): string {
  return CRITICALITY_COLOR[level?.toLowerCase() ?? ""] ?? "bg-zinc-500/15 text-zinc-400"
}

// Risk score: [0–1] where 0 = best, 1 = worst.
// Invert to get a 0–100 health score displayed as a bar.
function ScoreBar({ score, dim }: { score: number; dim: string }) {
  const pct = Math.round((1 - score) * 100)
  const barColor =
    pct >= 70 ? "bg-emerald-500" :
    pct >= 45 ? "bg-amber-500" :
    "bg-red-500"
  return (
    <div className="flex flex-col items-center gap-0.5 min-w-[36px]">
      <span className="text-[9px] font-bold uppercase tracking-wide text-muted-foreground">{dim}</span>
      <span className="text-xs font-bold tabular-nums leading-none">{pct}</span>
      <div className="w-full h-1 bg-muted/50 rounded-full overflow-hidden">
        <div className={cn("h-full rounded-full", barColor)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function CritBadge({ level }: { level: string | undefined }) {
  if (!level) return null
  return (
    <span className={cn("inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wide", critColor(level))}>
      {level}
    </span>
  )
}

// Type-specific icons using inline SVG paths to avoid bundle impact
function TypeIcon({ type }: { type: string }) {
  const cls = "h-3 w-3 shrink-0"
  switch (type) {
    case "Application":
      return <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>
    case "Node":
      return <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 7h10M7 12h10M7 17h10"/></svg>
    case "Topic":
      return <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
    case "Broker":
      return <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><circle cx="12" cy="12" r="10"/><path d="M12 2v20M2 12h20"/></svg>
    case "Library":
      return <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
    default:
      return <svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}><circle cx="12" cy="12" r="10"/></svg>
  }
}

// Human-readable byte formatting
function fmtBytes(n: number): string {
  if (n < 1024)        return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 ** 2).toFixed(2)} MB`
}

function fmtVal(key: string, val: unknown): string {
  const n = typeof val === "number" ? val : (typeof val === "string" && val !== "" && !isNaN(Number(val)) ? Number(val) : null)
  if (n !== null && (key === "size" || key === "message_size" || /bytes/.test(key))) return fmtBytes(n)
  if (typeof val === "boolean") return val ? "yes" : "no"
  if (n !== null && Number.isFinite(n)) {
    // [0–1] risk keys → show 4 decimals if small
    if (n < 10) return n.toFixed(4)
    return n.toLocaleString(undefined, { maximumFractionDigits: 2 })
  }
  return String(val ?? "—")
}

function MetricRow({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-muted-foreground shrink-0">{label}</span>
      <span className="font-mono text-foreground tabular-nums">
        {value}{unit && <span className="ml-1 text-[9px] opacity-60">{unit}</span>}
      </span>
    </div>
  )
}

// Select type-specific essential fields from raw properties
function TypeSpecificRows({ type, properties }: { type: string; properties: Record<string, unknown> }) {
  const rows: { key: string; label: string; unit?: string }[] = []

  const get = (k: string) => properties[k]
  const hasKey = (k: string) => get(k) !== undefined && get(k) !== null && get(k) !== ""

  switch (type) {
    case "Application":
      if (hasKey("weight"))                rows.push({ key: "weight",               label: "QoS Weight",    unit: "[0–1]" })
      if (hasKey("loc"))                   rows.push({ key: "loc",                  label: "LoC",           unit: "lines" })
      if (hasKey("cyclomatic_complexity")) rows.push({ key: "cyclomatic_complexity", label: "Cyclomatic CC", unit: "CC" })
      if (hasKey("instability_code"))      rows.push({ key: "instability_code",     label: "Instability",   unit: "[0–1]" })
      if (hasKey("lcom_norm"))             rows.push({ key: "lcom_norm",            label: "LCOM (norm)",   unit: "[0–1]" })
      break

    case "Topic":
      if (hasKey("message_size") || hasKey("payload_size_bytes") || hasKey("size")) {
        const k = hasKey("message_size") ? "message_size" : hasKey("payload_size_bytes") ? "payload_size_bytes" : "size"
        rows.push({ key: k, label: "Payload Size" })
      }
      if (hasKey("frequency"))   rows.push({ key: "frequency",   label: "Frequency",  unit: "Hz" })
      if (hasKey("deadline_ms")) rows.push({ key: "deadline_ms", label: "Deadline",   unit: "ms" })
      if (hasKey("queue_size"))  rows.push({ key: "queue_size",  label: "Queue Size", unit: "msgs" })
      if (hasKey("weight"))      rows.push({ key: "weight",      label: "QoS Weight", unit: "[0–1]" })
      break

    case "Node":
      if (hasKey("weight"))      rows.push({ key: "weight",      label: "Priority",   unit: "[0–1]" })
      if (hasKey("path_count"))  rows.push({ key: "path_count",  label: "Paths",      unit: "paths" })
      break

    case "Broker":
      if (hasKey("weight"))      rows.push({ key: "weight",      label: "QoS Weight", unit: "[0–1]" })
      if (hasKey("path_count"))  rows.push({ key: "path_count",  label: "Paths",      unit: "paths" })
      break

    case "Library":
      if (hasKey("weight"))                rows.push({ key: "weight",               label: "QoS Weight",   unit: "[0–1]" })
      if (hasKey("cyclomatic_complexity")) rows.push({ key: "cyclomatic_complexity", label: "Cyclomatic CC", unit: "CC" })
      if (hasKey("lcom_norm"))             rows.push({ key: "lcom_norm",            label: "LCOM (norm)",  unit: "[0–1]" })
      break
  }

  if (rows.length === 0) return null
  return (
    <>
      <div className="h-px bg-border/50 my-1.5" />
      <div className="space-y-1 text-[11px]">
        {rows.map(({ key, label, unit }) => (
          <MetricRow key={key} label={label} value={fmtVal(key, get(key))} unit={unit} />
        ))}
      </div>
    </>
  )
}

// ── Tooltip content ───────────────────────────────────────────────────────────

export function ItemTooltipContent({ data }: { data: ItemTooltipData }) {
  const { type, scores, criticality_levels, criticality_level, metrics, properties } = data
  const overallLevel = criticality_levels?.overall ?? criticality_level

  return (
    <div className="min-w-[180px] max-w-[260px] space-y-1.5 text-[11px]">
      {/* Header: type + overall criticality */}
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-foreground font-semibold">
          <TypeIcon type={type} />
          <span>{type}</span>
        </div>
        {overallLevel && <CritBadge level={overallLevel} />}
      </div>

      {/* RMAV score bars */}
      {scores && (
        <>
          <div className="h-px bg-border/50 my-1.5" />
          <div className="flex items-end gap-2 justify-between">
            <ScoreBar score={scores.reliability}    dim="R" />
            <ScoreBar score={scores.maintainability} dim="M" />
            <ScoreBar score={scores.availability}   dim="A" />
            <ScoreBar score={scores.vulnerability}  dim="V" />
          </div>
          {criticality_levels && (
            <div className="flex gap-1.5 justify-between">
              {(["reliability","maintainability","availability","vulnerability"] as const).map(d => (
                <div key={d} className="flex-1 flex justify-center">
                  <CritBadge level={criticality_levels[d]} />
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {/* Structural metrics (bottleneck table context) */}
      {metrics && Object.keys(metrics).length > 0 && (
        <>
          <div className="h-px bg-border/50 my-1.5" />
          <div className="space-y-1">
            {metrics.weight !== undefined && metrics.weight !== null &&
              <MetricRow label="QoS Weight"   value={fmtVal("weight", metrics.weight)}      unit="[0–1]" />}
            {metrics.cascade_depth !== undefined && metrics.cascade_depth !== null &&
              <MetricRow label="Cascade Depth" value={fmtVal("cascade_depth", metrics.cascade_depth)} unit="hops" />}
            {metrics.pubsub_betweenness !== undefined && metrics.pubsub_betweenness !== null &&
              <MetricRow label="PubSub BT"    value={fmtVal("pubsub_betweenness", metrics.pubsub_betweenness)} />}
            {metrics.is_articulation_point === true &&
              <div className="flex items-center gap-1.5"><span className="text-red-500 font-bold">🔴 SPOF</span><span className="text-muted-foreground">Articulation point</span></div>}
            {metrics.is_directed_ap === true && metrics.is_articulation_point !== true &&
              <div className="flex items-center gap-1.5"><span className="text-orange-500 font-bold">🟠 Directed AP</span></div>}
          </div>
        </>
      )}

      {/* Type-specific raw properties */}
      {properties && <TypeSpecificRows type={type} properties={properties} />}
    </div>
  )
}

// ── Public component ──────────────────────────────────────────────────────────

/**
 * Wraps any inline element (component name, badge, etc.) with a hover tooltip
 * that shows essential fields for a specific item type (Application, Node,
 * Topic, Broker, Library).
 *
 * @example
 * <ItemTooltip data={{ type: "Application", scores: component.scores, criticality_levels: component.criticality_levels }}>
 *   <span className="truncate">{component.name}</span>
 * </ItemTooltip>
 */
export function ItemTooltip({ data, children, side = "top", className }: ItemTooltipProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className={cn("cursor-help", className)}>
          {children}
        </span>
      </TooltipTrigger>
      <TooltipContent side={side} className="p-2.5 leading-relaxed shadow-xl border-border/60 bg-popover text-popover-foreground">
        <ItemTooltipContent data={data} />
      </TooltipContent>
    </Tooltip>
  )
}
