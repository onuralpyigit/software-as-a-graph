"use client"

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Keyboard } from "lucide-react"

interface KeyboardShortcutsHelpProps {
  className?: string
  inline?: boolean
}

export function KeyboardShortcutsHelp({ className = "bottom-4 left-4", inline = false }: KeyboardShortcutsHelpProps) {
  const [isOpen, setIsOpen] = useState(false)

  const shortcuts = [
    { key: 'F / Home', description: 'Fit graph to view' },
    { key: '3 / D', description: 'Toggle 2D/3D view' },
    { key: 'R / F5', description: 'Refresh graph data' },
    { key: 'Ctrl+S', description: 'Export as PNG' },
    { key: 'Esc', description: 'Clear selection' },
  ]

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        className={`${inline ? '' : `absolute ${className} `}z-50 rounded-full shadow-lg bg-background/90 backdrop-blur-sm border border-border hover:bg-accent hover:scale-110 hover:shadow-xl transition-all duration-200`}
        onClick={() => setIsOpen(!isOpen)}
        title="Keyboard shortcuts"
      >
        <Keyboard className="h-4 w-4 transition-transform duration-200" />
      </Button>

      {isOpen && (
        <div className={`absolute ${inline ? 'bottom-12 left-0' : `${className.includes('left-') ? className.replace(/left-\S+/, 'left-4') : 'left-4'} ${className.includes('bottom-') ? className.replace(/bottom-\S+/, 'bottom-16') : 'bottom-16'}`} z-50 bg-background/95 backdrop-blur-md border border-border rounded-lg p-4 shadow-2xl max-w-sm animate-in slide-in-from-bottom-2 fade-in duration-200`}>
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-sm">Keyboard Shortcuts</h3>
            <button
              onClick={() => setIsOpen(false)}
              className="text-muted-foreground hover:text-foreground transition-colors duration-150 hover:scale-110"
            >
              ✕
            </button>
          </div>
          <div className="space-y-2">
            {shortcuts.map((shortcut, index) => (
              <div key={index} className="flex justify-between items-center text-xs hover:bg-accent/50 rounded px-2 py-1 transition-colors duration-150">
                <kbd className="px-2 py-1 bg-muted rounded border border-border font-mono shadow-sm">
                  {shortcut.key}
                </kbd>
                <span className="text-muted-foreground ml-3">{shortcut.description}</span>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-3 border-t border-border text-xs text-muted-foreground">
            <p className="leading-relaxed">💡 Click nodes to explore • Drag to reposition • Scroll to zoom</p>
          </div>
        </div>
      )}
    </>
  )
}
