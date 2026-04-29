import axios, { AxiosInstance } from 'axios'
import { API_BASE_URL } from '@/lib/config/api'
import { apiClient } from './client'

// ============================================================================
// Types
// ============================================================================

export interface TopicInfo {
  id: string
  name: string
  weight: number
  publisher_count: number
  subscriber_count: number
  broker_ids: string[]
  broker_names: string[]
  qos_reliability: string | null
  qos_durability: string | null
  qos_transport_priority: string | null
  size: number
}

export interface AppInfo {
  id: string
  name: string
  weight: number
  pub_topic_ids: string[]
  sub_topic_ids: string[]
}

export interface TopicParams {
  frequency_hz: number
  duration_sec: number
}

export interface TrafficTopicMetrics {
  topic_id: string
  topic_name: string
  weight: number
  publisher_count: number
  subscriber_count: number
  broker_ids: string[]
  broker_names: string[]
  // Effective parameters used for this topic
  frequency_hz: number
  duration_sec: number
  message_size_bytes: number
  msgs_published_per_sec: number
  msgs_delivered_per_sec: number
  msgs_total_per_sec: number
  msgs_published_total: number
  msgs_delivered_total: number
  bandwidth_in_bps: number
  bandwidth_out_bps: number
  bandwidth_total_bps: number
}

export interface BrokerUsageMetrics {
  broker_id: string
  broker_name: string
  topics_routed: string[]
  msgs_inbound_per_sec: number
  msgs_outbound_per_sec: number
  msgs_total_per_sec: number
  bandwidth_bps: number
  bandwidth_mbps: number
}

export interface TrafficSummary {
  selected_topics: number
  topics_found: number
  frequency_hz: number
  duration_sec: number
  message_size_bytes: number
  total_msgs_published: number
  total_msgs_delivered: number
  total_network_bps: number
  total_network_mbps: number
  total_network_kbps: number
  peak_topic_bps: number
  brokers_involved: number
}

export interface TrafficSimulationResult {
  summary: TrafficSummary
  per_topic: TrafficTopicMetrics[]
  broker_usage: BrokerUsageMetrics[]
}

export interface TrafficSimulationRequest {
  topic_ids: string[]
  frequency_hz: number
  duration_sec: number
  message_size_bytes: number
  per_topic_params?: Record<string, TopicParams>
}

// ============================================================================
// Traffic API Client
// ============================================================================

class TrafficAPI {
  private client: AxiosInstance

  constructor(baseURL: string = API_BASE_URL) {
    this.client = axios.create({
      baseURL,
      headers: { 'Content-Type': 'application/json' },
      timeout: 60000,
    })
  }

  private getCredentials() {
    const credentials = apiClient.getCredentials()
    if (!credentials) {
      throw new Error('No Neo4j credentials set. Please connect to the database first.')
    }
    return { credentials }
  }

  async listTopics(): Promise<TopicInfo[]> {
    const body = this.getCredentials()
    const response = await this.client.post('/api/v1/traffic/topics', body)
    if (!response.data.success) {
      throw new Error(response.data.message || 'Failed to fetch topics')
    }
    return response.data.topics as TopicInfo[]
  }

  async listApps(): Promise<AppInfo[]> {
    const body = this.getCredentials()
    const response = await this.client.post('/api/v1/traffic/apps', body)
    if (!response.data.success) {
      throw new Error(response.data.message || 'Failed to fetch apps')
    }
    return response.data.apps as AppInfo[]
  }

  async simulate(request: TrafficSimulationRequest): Promise<TrafficSimulationResult> {
    const body = { ...this.getCredentials(), ...request }
    const response = await this.client.post('/api/v1/traffic/simulate', body)
    if (!response.data.success) {
      throw new Error(response.data.message || 'Traffic simulation failed')
    }
    return {
      summary: response.data.summary,
      per_topic: response.data.per_topic,
      broker_usage: response.data.broker_usage,
    }
  }
}

export const trafficClient = new TrafficAPI()
