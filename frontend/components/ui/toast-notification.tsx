"use client"

import { useEffect, useState } from "react"
import { AlertCircle, CheckCircle, X, WifiOff, Loader2 } from "lucide-react"
import { Button } from "./button"

export type ToastType = "success" | "error" | "warning" | "info" | "connection-error"

interface ToastNotificationProps {
  message: string
  type: ToastType
  duration?: number
  onClose?: () => void
  actionLabel?: string
  onAction?: () => void
  actionLoading?: boolean
}

export function ToastNotification({
  message,
  type,
  duration = 0,
  onClose,
  actionLabel,
  onAction,
  actionLoading = false,
}: ToastNotificationProps) {
  const [isVisible, setIsVisible] = useState(true)

  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => {
        setIsVisible(false)
        onClose?.()
      }, duration)
      return () => clearTimeout(timer)
    }
  }, [duration, onClose])

  if (!isVisible) return null

  const config = {
    success: {
      icon: CheckCircle,
      bgColor: "bg-green-50 dark:bg-green-950/30",
      borderColor: "border-green-200 dark:border-green-800",
      textColor: "text-green-800 dark:text-green-200",
      iconColor: "text-green-600 dark:text-green-400",
    },
    error: {
      icon: AlertCircle,
      bgColor: "bg-red-50 dark:bg-red-950/30",
      borderColor: "border-red-200 dark:border-red-800",
      textColor: "text-red-800 dark:text-red-200",
      iconColor: "text-red-600 dark:text-red-400",
    },
    warning: {
      icon: AlertCircle,
      bgColor: "bg-amber-50 dark:bg-amber-950/30",
      borderColor: "border-amber-200 dark:border-amber-800",
      textColor: "text-amber-800 dark:text-amber-200",
      iconColor: "text-amber-600 dark:text-amber-400",
    },
    info: {
      icon: AlertCircle,
      bgColor: "bg-blue-50 dark:bg-blue-950/30",
      borderColor: "border-blue-200 dark:border-blue-800",
      textColor: "text-blue-800 dark:text-blue-200",
      iconColor: "text-blue-600 dark:text-blue-400",
    },
    "connection-error": {
      icon: WifiOff,
      bgColor: "bg-red-50 dark:bg-red-950/30",
      borderColor: "border-red-200 dark:border-red-800",
      textColor: "text-red-800 dark:text-red-200",
      iconColor: "text-red-600 dark:text-red-400",
    },
  }[type]

  const Icon = config.icon

  return (
    <div
      className={`fixed top-4 right-4 z-50 max-w-md rounded-lg border ${config.borderColor} ${config.bgColor} shadow-lg animate-in slide-in-from-top-5 duration-300`}
    >
      <div className="flex items-start gap-3 p-4">
        <Icon className={`h-5 w-5 ${config.iconColor} flex-shrink-0 mt-0.5`} />
        <div className="flex-1">
          <p className={`text-sm font-medium ${config.textColor}`}>{message}</p>
        </div>
        <div className="flex items-center gap-2">
          {actionLabel && onAction && (
            <Button
              variant="outline"
              size="sm"
              onClick={onAction}
              disabled={actionLoading}
              className="text-xs h-7"
            >
              {actionLoading ? (
                <>
                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                  {actionLabel}
                </>
              ) : (
                actionLabel
              )}
            </Button>
          )}
          {onClose && (
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => {
                setIsVisible(false)
                onClose()
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
