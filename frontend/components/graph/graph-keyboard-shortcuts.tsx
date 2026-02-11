"use client"

import { useEffect } from 'react'

interface GraphKeyboardShortcutsProps {
  onFitGraph?: () => void
  onToggle3D?: () => void
  onRefresh?: () => void
  onExportPNG?: () => void
  onClearSelection?: () => void
}

export function GraphKeyboardShortcuts({
  onFitGraph,
  onToggle3D,
  onRefresh,
  onExportPNG,
  onClearSelection,
}: GraphKeyboardShortcutsProps) {
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in input fields
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return
      }

      // Fit graph to view: F or Home
      if ((e.key === 'f' || e.key === 'F' || e.key === 'Home') && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        onFitGraph?.()
      }

      // Toggle 3D: 3 or D
      if ((e.key === '3' || e.key === 'd' || e.key === 'D') && !e.ctrlKey && !e.metaKey) {
        e.preventDefault()
        onToggle3D?.()
      }

      // Refresh: R or F5
      if (((e.key === 'r' || e.key === 'R') && !e.ctrlKey && !e.metaKey) || e.key === 'F5') {
        e.preventDefault()
        onRefresh?.()
      }

      // Export PNG: Ctrl/Cmd + S
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        onExportPNG?.()
      }

      // Clear selection: Escape
      if (e.key === 'Escape') {
        e.preventDefault()
        onClearSelection?.()
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [onFitGraph, onToggle3D, onRefresh, onExportPNG, onClearSelection])

  return null
}
