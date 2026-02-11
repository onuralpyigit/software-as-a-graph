import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Analysis - Genieus",
  description: "Quality analysis and architecture insights",
}

export default function AnalysisLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
