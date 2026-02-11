import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Graph - Genieus",
  description: "Interactive graph visualization",
}

export default function GraphLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
