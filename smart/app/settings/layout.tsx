import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Settings - SMART",
  description: "Application settings and configuration",
}

export default function SettingsLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
