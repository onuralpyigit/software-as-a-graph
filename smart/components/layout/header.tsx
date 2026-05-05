"use client"

import { ReactNode } from "react"
import { ThemeToggle } from "@/components/theme-toggle"

interface HeaderProps {
  title?: string
  description?: string
  actions?: ReactNode
}

export function Header({ title, description, actions }: HeaderProps) {
  return (
    <header className="flex h-16 items-center justify-between border-b bg-gradient-to-r from-slate-50 via-white to-slate-50 dark:from-slate-950 dark:via-background dark:to-slate-950 px-6 gap-4">
      {title && (
        <div className="flex items-center gap-3 flex-shrink-0">
          <div className="h-8 w-1 rounded-full bg-gradient-to-b from-cyan-400 to-blue-600" />
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-cyan-500 to-blue-700 dark:from-cyan-400 dark:to-blue-500 bg-clip-text text-transparent">
              {title}
            </h1>
            {description && (
              <p className="text-xs text-muted-foreground">{description}</p>
            )}
          </div>
        </div>
      )}
      {actions && (
        <div className="flex-1 max-w-2xl">
          {actions}
        </div>
      )}
      <ThemeToggle />
    </header>
  )
}
