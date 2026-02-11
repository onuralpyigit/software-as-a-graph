import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Statistics - Genieus",
  description: "Advanced statistical analysis and insights",
}

export default function StatisticsLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
