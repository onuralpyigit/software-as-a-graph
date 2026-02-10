"use client"

import { useEffect, useState } from "react"
import { useConnection } from "@/lib/stores/connection-store"
import { ToastNotification } from "@/components/ui/toast-notification"

export function ConnectionStatusBanner() {
  const { status, error, connect, config } = useConnection()
  const [dismissed, setDismissed] = useState(false)
  const [isReconnecting, setIsReconnecting] = useState(false)

  // Reset dismissed state when status changes
  useEffect(() => {
    setDismissed(false)
  }, [status, error])

  // Don't show banner if dismissed or if status is connected/disconnected
  if (dismissed || status === 'connected' || status === 'disconnected') {
    return null
  }

  const handleReconnect = async () => {
    if (!config) return
    setIsReconnecting(true)
    try {
      await connect(config)
    } catch (err) {
      console.error('Reconnection failed:', err)
    } finally {
      setIsReconnecting(false)
    }
  }

  if (status === 'error') {
    const message = config?.database
      ? `${error || 'Connection lost'} - Database: ${config.database}`
      : error || 'Connection lost'

    return (
      <ToastNotification
        type="connection-error"
        message={message}
        onClose={() => setDismissed(true)}
        actionLabel="Reconnect"
        onAction={config ? handleReconnect : undefined}
        actionLoading={isReconnecting}
      />
    )
  }

  return null
}
