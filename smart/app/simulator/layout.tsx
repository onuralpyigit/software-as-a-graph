import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Simulator - Genieus",
  description: "Estimate pub-sub network and broker load for selected topics",
}

export default function SimulatorLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
