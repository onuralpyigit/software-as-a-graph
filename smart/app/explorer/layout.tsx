import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Explorer - Genieus",
  description: "Explore and visualize the system graph",
}

export default function ExplorerLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
