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
    vulnerability: string
    overall: string
  }
  scores: {
    reliability: number
    maintainability: number
    availability: number
    vulnerability: number
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
    vulnerability: number
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
const MAX_CACHE_ITEMS = 3 // Limit number of cached analyses

// Use sessionStorage instead of localStorage - larger quota and persists during session
const storage = typeof window !== 'undefined' ? sessionStorage : null

// Helper to compress data by keeping only essential fields
const compressAnalysisResult = (result: AnalysisResult): AnalysisResult => {
  return {
    context: result.context,
    summary: result.summary,
    stats: result.stats,
    components: result.components || [],
    edges: result.edges || [],
    problems: result.problems || []
  }
}

// Safe storage operations with error handling
const saveToStorage = (cache: Record<string, AnalysisResult>) => {
  if (!storage) return

  try {
    // Keep only the most recent MAX_CACHE_ITEMS
    const keys = Object.keys(cache)
    if (keys.length > MAX_CACHE_ITEMS) {
      const keysToKeep = keys.slice(-MAX_CACHE_ITEMS)
      const trimmedCache: Record<string, AnalysisResult> = {}
      keysToKeep.forEach(key => {
        trimmedCache[key] = cache[key]
      })
      storage.setItem(STORAGE_KEY, JSON.stringify(trimmedCache))
    } else {
      storage.setItem(STORAGE_KEY, JSON.stringify(cache))
    }
  } catch (error: any) {
    // If quota exceeded, clear everything and just keep in memory
    if (error.name === 'QuotaExceededError') {
      console.warn('Storage quota exceeded, keeping analysis in memory only')
      storage.removeItem(STORAGE_KEY)
    } else {
      console.error('Failed to save analysis cache:', error)
    }
  }
}

const loadFromStorage = (): Record<string, AnalysisResult> => {
  if (!storage) return {}

  try {
    const saved = storage.getItem(STORAGE_KEY)
    if (saved) {
      return JSON.parse(saved)
    }
  } catch (error) {
    console.error('Failed to load analysis cache:', error)
    // Clear corrupted data
    storage.removeItem(STORAGE_KEY)
  }
  return {}
}

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AnalysisState>({
    cache: {}
  })

  // Load cache from localStorage on mount
  useEffect(() => {
    const loadedCache = loadFromStorage()
    if (Object.keys(loadedCache).length > 0) {
      setState({ cache: loadedCache })
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
      if (storage) storage.removeItem(STORAGE_KEY)
    }
  }

  const clearAll = () => {
    setState({ cache: {} })
    if (storage) storage.removeItem(STORAGE_KEY)
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
