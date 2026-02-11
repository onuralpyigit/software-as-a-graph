"use client"

import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Database, Settings, Info, CheckCircle2, ArrowRight } from "lucide-react"

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

  const steps = [
    { text: "Configure your Neo4j database connection", icon: "1" },
    { text: "Ensure your database is running and accessible", icon: "2" },
    { text: "Verify the connection credentials are correct", icon: "3" }
  ]

  return (
    <div className="w-full">
      <Card className="border-2 border-purple-500/50 dark:border-purple-500/50 bg-white/95 dark:bg-black/95 backdrop-blur-md shadow-2xl shadow-purple-500/20 hover:shadow-purple-500/30 hover:border-purple-500/70 transition-all duration-300 overflow-hidden">
        {/* Decorative top border */}
        <div className="h-1 w-full bg-gradient-to-r from-purple-500 via-pink-500 to-purple-500" />
        
        <CardHeader className="pb-6 pt-8 px-8">
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-5">
            {/* Icon with animated gradient */}
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl blur-xl opacity-30 group-hover:opacity-50 transition-opacity duration-300" />
              <div className="relative rounded-2xl bg-gradient-to-br from-purple-500/20 to-purple-600/20 dark:from-purple-500/30 dark:to-purple-600/30 p-4 ring-1 ring-purple-500/30 group-hover:ring-purple-500/50 transition-all duration-300">
                <Database className="h-8 w-8 text-purple-600 dark:text-purple-400 group-hover:scale-110 transition-transform duration-300" />
              </div>
            </div>
            
            {/* Title section */}
            <div className="flex-1 space-y-1.5">
              <CardTitle className="text-2xl font-bold tracking-tight">{title}</CardTitle>
              <CardDescription className="text-base text-muted-foreground">
                {description}
              </CardDescription>
            </div>
            
            {/* CTA Button - Desktop */}
            {showButton && (
              <Button
                onClick={() => router.push('/settings')}
                size="lg"
                className="hidden sm:flex ml-auto bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-300 group"
              >
                <Settings className="mr-2 h-4 w-4 group-hover:rotate-90 transition-transform duration-300" />
                Connect Now
                <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform duration-300" />
              </Button>
            )}
          </div>
        </CardHeader>

        <CardContent className="px-8 pb-8 space-y-6">
          {/* Information box with steps */}
          <div className="rounded-2xl bg-gradient-to-br from-muted/40 via-muted/20 to-muted/10 border border-border/40 p-6 space-y-5">
            <div className="flex items-center gap-2.5">
              <div className="rounded-xl bg-gradient-to-br from-blue-500/20 to-blue-600/20 dark:from-blue-500/30 dark:to-blue-600/30 p-2.5 ring-1 ring-blue-500/30">
                <Info className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <h3 className="font-semibold text-base text-foreground">Quick Setup Guide</h3>
            </div>
            
            {/* Step-by-step list */}
            <div className="space-y-3 pl-1">
              {steps.map((step, index) => (
                <div 
                  key={index}
                  className="flex items-start gap-3.5 group/item"
                >
                  <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 dark:from-purple-500/25 dark:to-purple-600/25 flex items-center justify-center ring-1 ring-purple-500/30 group-hover/item:ring-purple-500/50 transition-all duration-200">
                    <span className="text-xs font-bold text-purple-600 dark:text-purple-400">{step.icon}</span>
                  </div>
                  <p className="text-sm text-muted-foreground leading-relaxed pt-0.5 group-hover/item:text-foreground transition-colors duration-200">
                    {step.text}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Bottom CTA Buttons */}
          {showButton && (
            <div className="flex flex-col sm:flex-row gap-3 pt-2">
              <Button
                onClick={() => router.push('/settings')}
                size="lg"
                className="flex-1 sm:hidden bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white shadow-lg hover:shadow-xl hover:scale-105 transition-all duration-300 group"
              >
                <Settings className="mr-2 h-4 w-4 group-hover:rotate-90 transition-transform duration-300" />
                Connect Now
                <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform duration-300" />
              </Button>
              
              <Button
                onClick={() => router.push('/settings')}
                variant="outline"
                size="lg"
                className="flex-1 bg-white/10 dark:bg-white/10 text-foreground border-purple-500/30 hover:bg-purple-500/10 hover:border-purple-500/50 backdrop-blur-sm transition-all duration-300 group"
              >
                <Settings className="mr-2 h-4 w-4 group-hover:rotate-90 transition-transform duration-300" />
                Go to Settings
              </Button>
              
              <Button
                onClick={() => router.push('/tutorial')}
                variant="outline"
                size="lg"
                className="flex-1 bg-white/10 dark:bg-white/10 text-foreground border-purple-500/30 hover:bg-purple-500/10 hover:border-purple-500/50 backdrop-blur-sm transition-all duration-300 group"
              >
                <Info className="mr-2 h-4 w-4" />
                View Tutorial
                <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform duration-300" />
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
