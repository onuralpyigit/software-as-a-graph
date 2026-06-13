import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Statistics - SMART",
  description: "Structural and communication statistics for a pub-sub system: topic bandwidth, fanout, node load, domain coupling, critical-component ratios, and library dependency density.",
}

export default function StatisticsLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
