"use client"

import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Database, Settings } from "lucide-react"

interface NoConnectionInfoProps {
  title?: string
  description?: string
  showButton?: boolean
}

export function NoConnectionInfo({
  title = "No Database Connection",
  description = "Connect to your Neo4j database to get started",
  showButton = true
}: NoConnectionInfoProps) {
  const router = useRouter()

  return (
    <div className="flex flex-col items-center justify-center py-20 gap-5 text-center">
      <div className="rounded-full bg-muted p-4">
        <Database className="h-7 w-7 text-muted-foreground" />
      </div>
      <div className="space-y-1.5 max-w-sm">
        <p className="text-base font-semibold text-foreground">{title}</p>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      {showButton && (
        <Button
          onClick={() => router.push('/settings')}
          variant="outline"
          size="sm"
          className="mt-1"
        >
          <Settings className="mr-2 h-4 w-4" />
          Go to Settings
        </Button>
      )}
    </div>
  )
}
