"use client"

import { ReactNode } from "react"
import { Sidebar } from "./sidebar"
import { Header } from "./header"
import { ConnectionStatusBanner } from "./connection-status-banner"

interface AppLayoutProps {
  children: ReactNode
  title?: string
  description?: string
  actions?: ReactNode
}

export function AppLayout({ children, title, description, actions }: AppLayoutProps) {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Header */}
        <Header title={title} description={description} actions={actions} />

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-6">
          {children}
        </main>
      </div>

      {/* Global Connection Status Toast */}
      <ConnectionStatusBanner />
    </div>
  )
}
