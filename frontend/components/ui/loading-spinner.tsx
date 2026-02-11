import { Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface LoadingSpinnerProps {
  size?: "sm" | "md" | "lg"
  text?: string
  className?: string
}

export function LoadingSpinner({ size = "md", text, className }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: "h-5 w-5",
    md: "h-10 w-10",
    lg: "h-16 w-16",
  }

  const textSizeClasses = {
    sm: "text-xs",
    md: "text-sm",
    lg: "text-base",
  }

  return (
    <div className={cn("flex flex-col items-center justify-center gap-3", className)}>
      <div className="relative">
        {/* Single spinning ring with gradient */}
        <div
          className={cn(
            "rounded-full border-2 border-slate-200 dark:border-slate-800",
            sizeClasses[size]
          )}
        />
        <div
          className={cn(
            "absolute inset-0 rounded-full border-2 border-transparent border-t-blue-500 border-r-purple-500 animate-spin",
            sizeClasses[size]
          )}
        />
      </div>
      {text && (
        <p className={cn("font-medium text-muted-foreground", textSizeClasses[size])}>
          {text}
        </p>
      )}
    </div>
  )
}
