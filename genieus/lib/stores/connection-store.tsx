"use client"

import { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { apiClient } from '@/lib/api/client'
import type { Neo4jConfig, GraphStatsResponse } from '@/lib/types/api'

interface ConnectionState {
  status: 'connected' | 'disconnected' | 'error' | 'connecting'
  config?: Neo4jConfig
  stats?: GraphStatsResponse
  error?: string
  initialLoadComplete?: boolean
}

interface ConnectionContextType extends ConnectionState {
  connect: (config: Neo4jConfig) => Promise<void>
  disconnect: () => void
  refreshStats: () => Promise<void>
}

const ConnectionContext = createContext<ConnectionContextType | undefined>(undefined)

const STORAGE_KEY = 'neo4j-connection-config'

export function ConnectionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<ConnectionState>({
    status: 'connecting',
    initialLoadComplete: false,
  })

  // Setup connection error handler on mount
  useEffect(() => {
    // Register error handler with API client
    apiClient.setConnectionErrorHandler(() => {
      setState(prev => {
        // Only update if currently connected
        if (prev.status === 'connected') {
          return {
            ...prev,
            status: 'error',
            error: 'Connection lost',
          }
        }
        return prev
      })
    })
  }, [])

  // Load saved config on mount
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      try {
        const config = JSON.parse(saved) as Neo4jConfig
        // Auto-connect on load, but don't block if it fails
        connect(config).catch((error) => {
          console.error('Auto-connect failed:', error)
          // Don't show error on auto-connect failure, just set to disconnected
          setState(prev => ({
            ...prev,
            status: 'disconnected',
            error: undefined,
            initialLoadComplete: true
          }))
        })
      } catch (e) {
        console.error('Failed to load saved connection config:', e)
        setState(prev => ({ ...prev, status: 'disconnected', initialLoadComplete: true }))
      }
    } else {
      // No saved config, automatically try default credentials
      const getDefaultNeo4jUri = () => {
        if (typeof window !== 'undefined') {
          const hostname = window.location.hostname;
          return `bolt://${hostname}:7687`;
        }
        return "bolt://localhost:7687";
      };

      const defaultConfig: Neo4jConfig = {
        uri: getDefaultNeo4jUri(),
        user: "neo4j",
        password: "password",
        database: "neo4j",
      }
      
      // Auto-connect with default password, but don't show error if it fails
      connect(defaultConfig).catch((error) => {
        console.log('Auto-connect with default password failed, user needs to configure:', error)
        // Don't show error on auto-connect failure, just set to disconnected
        setState(prev => ({
          ...prev,
          status: 'disconnected',
          error: undefined,
          initialLoadComplete: true
        }))
      })
    }
  }, [])

  const connect = async (config: Neo4jConfig) => {
    setState(prev => ({ ...prev, status: 'connecting', error: undefined }))

    try {
      // Update API client base URL if needed
      // Use dynamic hostname if accessing from remote
      let baseURL = localStorage.getItem('api-base-url');
      if (!baseURL && typeof window !== 'undefined') {
        // Auto-detect: use same hostname as frontend with port 8000
        const hostname = window.location.hostname;
        const protocol = window.location.protocol;
        baseURL = `${protocol}//${hostname}:8000`;
      }
      if (!baseURL) {
        baseURL = 'http://localhost:8000';
      }
      apiClient.setBaseURL(baseURL)

      // Attempt connection
      const response = await apiClient.connect(config)

      // Fetch initial stats
      const stats = await apiClient.getGraphStats()

      // Save config to localStorage (without password for security)
      const configToSave = { ...config }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(configToSave))

      setState({
        status: 'connected',
        config,
        stats,
        error: undefined,
        initialLoadComplete: true,
      })
    } catch (error: any) {
      // Better error message for network errors
      let errorMessage = 'Connection failed'

      if (error.code === 'ERR_NETWORK' || error.message?.includes('Network Error')) {
        errorMessage = 'Cannot reach analysis service. Ensure the API is connected first.'
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail
      } else if (error.message) {
        errorMessage = error.message
      }

      setState({
        status: 'error',
        error: errorMessage,
        initialLoadComplete: true,
      })
      throw error
    }
  }

  const disconnect = () => {
    localStorage.removeItem(STORAGE_KEY)
    apiClient.clearCredentials()  // Clear credentials from API client
    setState({
      status: 'disconnected',
      config: undefined,
      stats: undefined,
      error: undefined,
      initialLoadComplete: true,
    })
  }

  const refreshStats = async () => {
    if (state.status !== 'connected') return

    try {
      const stats = await apiClient.getGraphStats()
      setState(prev => ({ ...prev, stats }))
    } catch (error) {
      console.error('Failed to refresh stats:', error)
    }
  }

  return (
    <ConnectionContext.Provider
      value={{
        ...state,
        connect,
        disconnect,
        refreshStats,
      }}
    >
      {children}
    </ConnectionContext.Provider>
  )
}

export function useConnection() {
  const context = useContext(ConnectionContext)
  if (context === undefined) {
    throw new Error('useConnection must be used within a ConnectionProvider')
  }
  return context
}
