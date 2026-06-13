import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Data - SMART",
  description: "Data management and graph generation",
}

export default function DataLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
