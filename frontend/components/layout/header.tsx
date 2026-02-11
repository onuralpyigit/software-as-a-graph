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
          <div className="h-8 w-1 rounded-full bg-gradient-to-b from-blue-500 to-purple-600" />
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent">
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
