"use client"

import React from "react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { getScoreRangeDescription, ScoreType } from "@/lib/score-ranges"
import { cn } from "@/lib/utils"

interface ScoreTooltipProps {
  /** The numeric score value — used to determine the range description. */
  score: number
  /** The type of score — determines which range interpretation is shown. */
  type: ScoreType
  /** The displayed content (typically the formatted number string). */
  children: React.ReactNode
  /** Tooltip placement. Defaults to "top". */
  side?: "top" | "right" | "bottom" | "left"
  /** Extra class names applied to the trigger wrapper span. */
  className?: string
}

/**
 * Wraps a numerical score with a hover tooltip that explains what the value
 * means within its expected range.
 *
 * Usage:
 *   <ScoreTooltip score={72.4} type="quality">72</ScoreTooltip>
 *   <ScoreTooltip score={0.91} type="spearman">{value.toFixed(4)}</ScoreTooltip>
 */
export function ScoreTooltip({
  score,
  type,
  children,
  side = "top",
  className,
}: ScoreTooltipProps) {
  const description = getScoreRangeDescription(score, type)
  if (!description) return <>{children}</>

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={cn(
            "border-b border-dashed border-current/40 cursor-help",
            className
          )}
        >
          {children}
        </span>
      </TooltipTrigger>
      <TooltipContent side={side} className="max-w-64 text-xs leading-relaxed">
        {description}
      </TooltipContent>
    </Tooltip>
  )
}
