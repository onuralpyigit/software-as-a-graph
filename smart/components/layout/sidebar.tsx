"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState } from "react"
import { cn } from "@/lib/utils"
import Image from "next/image"
import {
  LayoutDashboard,
  FileText,
  Settings,
  Database,
  Network,
  Zap,
  BarChart3,
  ShieldCheck,
  Brain,
  Cpu,
  Activity,
  BookMarked,
} from "lucide-react"

const navigation = [
  {
    name: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    name: "Explorer",
    href: "/explorer",
    icon: Network,
  },
  {
    name: "Statistics",
    href: "/statistics",
    icon: BarChart3,
  },
  {
    name: "Simulator",
    href: "/simulator",
    icon: Activity,
  },
  {
    name: "Analysis",
    href: "/analysis",
    icon: FileText,
  },
  {
    name: "Simulation",
    href: "/simulation",
    icon: Zap,
  },
  {
    name: "Validation",
    href: "/validation",
    icon: ShieldCheck,
  },
  {
    name: "Train",
    href: "/train",
    icon: Brain,
  },
  {
    name: "Predict",
    href: "/predict",
    icon: Cpu,
  },
  {
    name: "Data",
    href: "/data",
    icon: Database,
  },
  {
    name: "Glossary",
    href: "/glossary",
    icon: BookMarked,
  },
  {
    name: "Settings",
    href: "/settings",
    icon: Settings,
  },
]

export function Sidebar() {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = useState(true)

  return (
    <div
      className={cn(
        "relative flex h-full flex-col border-r bg-gradient-to-b from-slate-50 via-white to-slate-50 dark:from-slate-950 dark:via-background dark:to-slate-950 transition-all duration-300",
        collapsed ? "w-16" : "w-64"
      )}
      onMouseEnter={() => setCollapsed(false)}
      onMouseLeave={() => setCollapsed(true)}
    >
      {/* Logo/Title */}
      <div className={cn(
        "flex h-16 items-center border-b bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-950/30 dark:to-blue-950/30",
        collapsed ? "justify-center px-0" : "px-6"
      )}>
        <Image
          src="/smart.png"
          alt="SMART Logo"
          width={32}
          height={32}
          className="rounded-lg shrink-0"
        />
        {!collapsed && (
          <span className="ml-3 text-lg font-bold bg-gradient-to-r from-cyan-500 to-blue-700 dark:from-cyan-400 dark:to-blue-500 bg-clip-text text-transparent whitespace-nowrap overflow-hidden">SMART</span>
        )}
      </div>

      {/* Navigation */}
      <nav className={cn("flex-1 space-y-1 p-2", !collapsed && "p-4")}>
        {navigation.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(item.href + "/")
          const Icon = item.icon

          return (
            <Link
              key={item.name}
              href={item.href}
              title={collapsed ? item.name : undefined}
              onClick={() => setCollapsed(true)}
              className={cn(
                "flex items-center rounded-lg px-3 py-2.5 text-sm font-medium transition-colors",
                collapsed ? "justify-center gap-0" : "gap-3",
                isActive
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
              )}
            >
              <Icon className="h-5 w-5 shrink-0" />
              {!collapsed && item.name}
            </Link>
          )
        })}
      </nav>
    </div>
  )
}
