import React from 'react'
import { ChevronRight } from 'lucide-react'

interface BreadcrumbNavigationProps {
  breadcrumbPath: Array<{id: string | null, label: string}>
  onBreadcrumbClick: (id: string | null) => void
  onReset: () => void
}

export function BreadcrumbNavigation({
  breadcrumbPath,
  onBreadcrumbClick,
  onReset
}: BreadcrumbNavigationProps) {
  if (breadcrumbPath.length <= 1) return null

  return (
    <div className="flex items-center gap-1 text-xs">
      {breadcrumbPath.map((crumb, index) => (
        <React.Fragment key={crumb.id || 'root'}>
          <button
            onClick={() => onBreadcrumbClick(crumb.id)}
            className={`px-2 py-1 rounded transition-colors ${
              index === breadcrumbPath.length - 1
                ? 'bg-primary/10 text-primary font-medium cursor-default'
                : 'hover:bg-accent text-muted-foreground hover:text-foreground'
            }`}
            title={index === breadcrumbPath.length - 1 ? `${crumb.label} (current)` : crumb.label}
            disabled={index === breadcrumbPath.length - 1}
          >
            <span className="truncate max-w-[100px] inline-block">{crumb.label}</span>
          </button>
          {index < breadcrumbPath.length - 1 && (
            <ChevronRight className="h-3 w-3 text-muted-foreground/50 flex-shrink-0" />
          )}
        </React.Fragment>
      ))}
    </div>
  )
}
