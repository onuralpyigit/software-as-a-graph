"use client"

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface ComponentAnalysis {
  id: string
  name: string
  type: string
  criticality_level: string
  criticality_levels?: {
    reliability: string
    maintainability: string
    availability: string
    security: string
    overall: string
  }
  scores: {
    reliability: number
    maintainability: number
    availability: number
    security: number
    overall: number
  }
}

interface EdgeAnalysis {
  source: string
  target: string
  source_name?: string
  target_name?: string
  type: string
  criticality_level: string
  scores: {
    reliability: number
    maintainability: number
    availability: number
    security: number
    overall: number
  }
}

interface Problem {
  entity_id: string
  type: string
  category: string
  severity: string
  name: string
  description: string
  recommendation: string
}

interface AnalysisResult {
  context?: string
  summary: any
  stats: any
  components: ComponentAnalysis[]
  edges?: EdgeAnalysis[]
  problems: Problem[]
  logs?: string[]
}

interface AnalysisState {
  cache: Record<string, AnalysisResult>
}

interface AnalysisContextType extends AnalysisState {
  setAnalysis: (key: string, result: AnalysisResult) => void
  getAnalysis: (key: string) => AnalysisResult | null
  clearAnalysis: (key?: string) => void
  clearAll: () => void
}

const AnalysisContext = createContext<AnalysisContextType | undefined>(undefined)

const STORAGE_KEY = 'analysis-cache'
const CACHE_VERSION = 3
const MAX_CACHE_ITEMS = 4 // one per layer: system, application, infrastructure, middleware

// Access localStorage lazily so this module works during SSR (window is undefined server-side).
// Never capture it at module evaluation time — Next.js evaluates modules on the server where
// window doesn't exist, and a captured null would persist into the browser runtime.
const getStorage = (): Storage | null => {
  if (typeof window === 'undefined') return null
  return window.localStorage
}

// Helper to compress data by keeping only essential fields
const compressAnalysisResult = (result: AnalysisResult): AnalysisResult => {
  return {
    context: result.context,
    summary: result.summary,
    stats: result.stats,
    components: result.components || [],
    edges: result.edges || [],
    problems: result.problems || [],
    logs: result.logs || [],
  }
}

// Safe storage operations with error handling
const saveToStorage = (cache: Record<string, AnalysisResult>) => {
  const storage = getStorage()
  if (!storage) return

  // Keep only the most recent MAX_CACHE_ITEMS
  const keys = Object.keys(cache)
  const trimmedCache: Record<string, AnalysisResult> = {}
  const keysToKeep = keys.length > MAX_CACHE_ITEMS ? keys.slice(-MAX_CACHE_ITEMS) : keys
  keysToKeep.forEach(key => { trimmedCache[key] = cache[key] })

  const tryWrite = (payload: object) => {
    storage.setItem(STORAGE_KEY, JSON.stringify({ v: CACHE_VERSION, data: payload }))
  }

  try {
    tryWrite(trimmedCache)
  } catch (e1: any) {
    if (e1.name !== 'QuotaExceededError') {
      console.error('Failed to save analysis cache:', e1)
      return
    }
    // Quota exceeded — strip logs from each entry and retry once
    try {
      const slim: Record<string, AnalysisResult> = {}
      Object.entries(trimmedCache).forEach(([k, v]) => {
        slim[k] = { ...v, logs: [] }
      })
      tryWrite(slim)
    } catch (e2: any) {
      console.warn('Storage quota exceeded even after stripping logs; cache not persisted')
      storage.removeItem(STORAGE_KEY)
    }
  }
}

const loadFromStorage = (): Record<string, AnalysisResult> => {
  const storage = getStorage()
  if (!storage) return {}

  try {
    const saved = storage.getItem(STORAGE_KEY)
    if (saved) {
      const parsed = JSON.parse(saved)
      if (parsed.v !== CACHE_VERSION) {
        storage.removeItem(STORAGE_KEY)
        return {}
      }
      return parsed.data as Record<string, AnalysisResult>
    }
  } catch (error) {
    console.error('Failed to load analysis cache:', error)
    storage.removeItem(STORAGE_KEY)
  }
  return {}
}

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AnalysisState>({ cache: {} })

  // Load from localStorage on mount (client-only — window is unavailable during SSR)
  useEffect(() => {
    const loaded = loadFromStorage()
    if (Object.keys(loaded).length > 0) {
      setState({ cache: loaded })
    }
  }, [])

  const setAnalysis = (key: string, result: AnalysisResult) => {
    setState(prev => {
      const newCache = {
        ...prev.cache,
        [key]: compressAnalysisResult(result)
      }
      // Save to localStorage after state update
      saveToStorage(newCache)
      return { cache: newCache }
    })
  }

  const getAnalysis = (key: string): AnalysisResult | null => {
    return state.cache[key] || null
  }

  const clearAnalysis = (key?: string) => {
    if (key) {
      setState(prev => {
        const newCache = { ...prev.cache }
        delete newCache[key]
        saveToStorage(newCache)
        return { cache: newCache }
      })
    } else {
      // Clear all
      setState({ cache: {} })
      getStorage()?.removeItem(STORAGE_KEY)
    }
  }

  const clearAll = () => {
    setState({ cache: {} })
    getStorage()?.removeItem(STORAGE_KEY)
  }

  return (
    <AnalysisContext.Provider
      value={{
        ...state,
        setAnalysis,
        getAnalysis,
        clearAnalysis,
        clearAll
      }}
    >
      {children}
    </AnalysisContext.Provider>
  )
}

export function useAnalysis() {
  const context = useContext(AnalysisContext)
  if (context === undefined) {
    throw new Error('useAnalysis must be used within an AnalysisProvider')
  }
  return context
}
