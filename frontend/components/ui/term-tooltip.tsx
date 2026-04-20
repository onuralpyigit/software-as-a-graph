"use client"

import React from "react"
import { Info } from "lucide-react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { TERM_TOOLTIPS } from "@/lib/tooltips"
import { cn } from "@/lib/utils"

interface TermTooltipProps {
  /** The term to look up in TERM_TOOLTIPS. Falls back to `children` text if not found. */
  term?: string
  /** Custom tooltip text — overrides the dictionary lookup. */
  description?: string
  /** Content to render — defaults to `term` if not provided. */
  children?: React.ReactNode
  /** Side the tooltip appears on. */
  side?: "top" | "right" | "bottom" | "left"
  /** Additional className for the trigger wrapper. */
  className?: string
  /** Show just the info icon without the term label. */
  iconOnly?: boolean
}

/**
 * Inline term with a subtle info icon that shows a tooltip on hover.
 *
 * Usage:
 *   <TermTooltip term="Spearman ρ" />
 *   <TermTooltip term="SPOF">Single Point of Failure</TermTooltip>
 *   <TermTooltip term="R(v)" iconOnly />
 */
export function TermTooltip({
  term,
  description,
  children,
  side = "top",
  className,
  iconOnly = false,
}: TermTooltipProps) {
  const tooltipText = description ?? (term ? TERM_TOOLTIPS[term] : undefined)

  // No description found — render children/term as plain text
  if (!tooltipText) {
    return <>{children ?? term}</>
  }

  if (iconOnly) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className={cn("inline-flex items-center cursor-help", className)}>
            <Info className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground transition-colors" />
          </span>
        </TooltipTrigger>
        <TooltipContent side={side} className="max-w-72 text-xs leading-relaxed">
          {tooltipText}
        </TooltipContent>
      </Tooltip>
    )
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className={cn(
            "inline-flex items-center gap-1 cursor-help underline decoration-dotted decoration-muted-foreground/50 underline-offset-2",
            className,
          )}
        >
          {children ?? term}
          <Info className="h-3 w-3 shrink-0 text-muted-foreground/60" />
        </span>
      </TooltipTrigger>
      <TooltipContent side={side} className="max-w-72 text-xs leading-relaxed">
        {tooltipText}
      </TooltipContent>
    </Tooltip>
  )
}
