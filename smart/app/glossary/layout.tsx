import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Glossary - SMART",
  description: "Key terms, metrics, and abbreviations used across the platform",
}

export default function GlossaryLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
