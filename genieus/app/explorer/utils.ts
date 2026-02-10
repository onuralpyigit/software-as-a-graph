import { Server, Cpu, MessageSquare, BookOpen, Network, HelpCircle } from 'lucide-react'
import type { GraphLink } from '@/lib/types/api'

export function getNodeIcon(type: string) {
  switch (type) {
    case 'Node': return Server
    case 'Application': return Cpu
    case 'Topic': return MessageSquare
    case 'Library': return BookOpen
    case 'Broker': return Network
    default: return HelpCircle
  }
}

export function getLinkId(link: GraphLink): string {
  const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
  const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
  return `${sourceId}-${link.type}-${targetId}`
}

export function getNodeColorMap(theme: string | undefined) {
  const isDark = theme === 'dark'
  return {
    Application: isDark ? "#3b82f6" : "#2563eb",
    Node: isDark ? "#ef4444" : "#dc2626",
    Broker: isDark ? "#a1a1aa" : "#71717a",
    Topic: isDark ? "#facc15" : "#eab308",
    Library: isDark ? "#06b6d4" : "#0891b2",
    Unknown: isDark ? "#a1a1aa" : "#71717a",
  }
}

export function getLinkColorMap(theme: string | undefined) {
  const isDark = theme === 'dark'
  return {
    RUNS_ON: isDark ? "#a855f7" : "#9333ea",
    PUBLISHES_TO: isDark ? "#22c55e" : "#16a34a",
    SUBSCRIBES_TO: isDark ? "#f97316" : "#ea580c",
    DEPENDS_ON: isDark ? "#ef4444" : "#dc2626",
    CONNECTS_TO: isDark ? "#22c55e" : "#16a34a",
    ROUTES: isDark ? "#a1a1aa" : "#71717a",
    USES: isDark ? "#06b6d4" : "#0891b2",
  }
}
