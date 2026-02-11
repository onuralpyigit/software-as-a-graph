import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "Tutorial - Genieus",
  description: "Learn how to use Genieus",
}

export default function TutorialLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return children
}
