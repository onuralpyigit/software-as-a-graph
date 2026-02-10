"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import Image from "next/image"
import {
  LayoutDashboard,
  Network,
  FileText,
  Settings,
  Database,
  BookOpen,
  Waypoints,
  Zap,
  BarChart3,
  ShieldCheck,
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
    name: "Analysis",
    href: "/analysis",
    icon: FileText,
  },
  {
    name: "Statistics",
    href: "/statistics",
    icon: BarChart3,
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
    name: "Data",
    href: "/data",
    icon: Database,
  },
  {
    name: "Tutorial",
    href: "/tutorial",
    icon: BookOpen,
  },
  {
    name: "Settings",
    href: "/settings",
    icon: Settings,
  },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <div className="flex h-full w-64 flex-col border-r bg-gradient-to-b from-slate-50 via-white to-slate-50 dark:from-slate-950 dark:via-background dark:to-slate-950">
      {/* Logo/Title */}
      <div className="flex h-16 items-center border-b px-6 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/30 dark:to-purple-950/30">
        <Image 
          src="/smart.png" 
          alt="Genieus Logo" 
          width={32} 
          height={32}
          className="rounded-lg shadow-lg"
        />
        <span className="ml-3 text-lg font-bold bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent">Genieus</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 p-4">
        {navigation.map((item) => {
          const isActive = pathname === item.href || pathname.startsWith(item.href + "/")
          const Icon = item.icon

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-md shadow-blue-500/20 dark:shadow-blue-500/30"
                  : "text-muted-foreground hover:bg-gradient-to-r hover:from-blue-50 hover:to-purple-50 dark:hover:from-blue-950/50 dark:hover:to-purple-950/50 hover:text-foreground hover:shadow-sm"
              )}
            >
              <Icon className="h-5 w-5" />
              {item.name}
            </Link>
          )
        })}
      </nav>
    </div>
  )
}
