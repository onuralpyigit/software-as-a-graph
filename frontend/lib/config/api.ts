/**
 * API Configuration
 * 
 * Configuration for the FastAPI backend endpoints.
 * Change API_BASE_URL to point to your FastAPI server.
 */

// Helper to get the default API URL based on the current hostname
function getDefaultApiUrl(): string {
  // If NEXT_PUBLIC_API_URL is set, use it
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  
  // For client-side, use the same hostname as the frontend but with port 8000
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;
    return `${protocol}//${hostname}:8000`;
  }
  
  // For server-side rendering, default to localhost
  return 'http://localhost:8000';
}

// Default: Use the new FastAPI server at port 8000
export const API_BASE_URL = getDefaultApiUrl();

// Alternative: Use the old API server (uncomment if needed)
// export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export const API_CONFIG = {
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minutes
  headers: {
    'Content-Type': 'application/json',
  },
};

// Neo4j Configuration (for reference, actual config in settings)
function getDefaultNeo4jUri(): string {
  // For client-side, use the same hostname as the frontend
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    return `bolt://${hostname}:7687`;
  }
  // For server-side rendering, default to localhost
  return 'bolt://localhost:7687';
}

export const NEO4J_DEFAULT_CONFIG = {
  uri: typeof window !== 'undefined' ? getDefaultNeo4jUri() : 'bolt://localhost:7687',
  user: 'neo4j',
  password: 'password',
  database: 'neo4j',
};
