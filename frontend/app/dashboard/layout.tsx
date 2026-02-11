import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Dashboard - Genieus",
  description: "System overview and statistics",
}

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
