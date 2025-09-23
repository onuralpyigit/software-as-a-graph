var graphData = {
  "nodes": [
    {
      "id": "App-1",
      "label": "App-1",
      "group": "Application",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "App-2",
      "label": "App-2",
      "group": "Application",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Has 5 dependent applications (56% of other applications)"
        ],
        "metrics": {
          "dependent_count": 5,
          "dependency_ratio": 0.5555555555555556,
          "exclusive_topics": 0,
          "exclusive_topic_names": []
        }
      }
    },
    {
      "id": "App-3",
      "label": "App-3",
      "group": "Application",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Has 5 dependent applications (56% of other applications)"
        ],
        "metrics": {
          "dependent_count": 5,
          "dependency_ratio": 0.5555555555555556,
          "exclusive_topics": 1,
          "exclusive_topic_names": [
            "Topic-7"
          ]
        }
      }
    },
    {
      "id": "App-4",
      "label": "App-4",
      "group": "Application",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Has 6 dependent applications (67% of other applications)"
        ],
        "metrics": {
          "dependent_count": 6,
          "dependency_ratio": 0.6666666666666666,
          "exclusive_topics": 1,
          "exclusive_topic_names": [
            "Topic-22"
          ]
        }
      }
    },
    {
      "id": "App-5",
      "label": "App-5",
      "group": "Application",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Has 6 dependent applications (67% of other applications)",
          "Sole publisher for 3 topics"
        ],
        "metrics": {
          "dependent_count": 6,
          "dependency_ratio": 0.6666666666666666,
          "exclusive_topics": 3,
          "exclusive_topic_names": [
            "Topic-15",
            "Topic-20",
            "Topic-25"
          ]
        }
      }
    },
    {
      "id": "App-6",
      "label": "App-6",
      "group": "Application",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Has 6 dependent applications (67% of other applications)",
          "Sole publisher for 3 topics"
        ],
        "metrics": {
          "dependent_count": 6,
          "dependency_ratio": 0.6666666666666666,
          "exclusive_topics": 3,
          "exclusive_topic_names": [
            "Topic-8",
            "Topic-10",
            "Topic-24"
          ]
        }
      }
    },
    {
      "id": "App-7",
      "label": "App-7",
      "group": "Application",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Has 8 dependent applications (89% of other applications)",
          "Sole publisher for 3 topics"
        ],
        "metrics": {
          "dependent_count": 8,
          "dependency_ratio": 0.8888888888888888,
          "exclusive_topics": 3,
          "exclusive_topic_names": [
            "Topic-5",
            "Topic-13",
            "Topic-21"
          ]
        }
      }
    },
    {
      "id": "App-8",
      "label": "App-8",
      "group": "Application",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Has 4 dependent applications (44% of other applications)"
        ],
        "metrics": {
          "dependent_count": 4,
          "dependency_ratio": 0.4444444444444444,
          "exclusive_topics": 0,
          "exclusive_topic_names": []
        }
      }
    },
    {
      "id": "App-9",
      "label": "App-9",
      "group": "Application",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Has 7 dependent applications (78% of other applications)",
          "Sole publisher for 2 topics"
        ],
        "metrics": {
          "dependent_count": 7,
          "dependency_ratio": 0.7777777777777778,
          "exclusive_topics": 2,
          "exclusive_topic_names": [
            "Topic-18",
            "Topic-19"
          ]
        }
      }
    },
    {
      "id": "App-10",
      "label": "App-10",
      "group": "Application",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Has 6 dependent applications (67% of other applications)",
          "Sole publisher for 2 topics"
        ],
        "metrics": {
          "dependent_count": 6,
          "dependency_ratio": 0.6666666666666666,
          "exclusive_topics": 2,
          "exclusive_topic_names": [
            "Topic-12",
            "Topic-17"
          ]
        }
      }
    },
    {
      "id": "Broker-1",
      "label": "Broker-1",
      "group": "Broker",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Routes 12 topics (48% of all topics)",
          "Impacts 10 applications (100% of all applications)"
        ],
        "metrics": {
          "topic_count": 12,
          "topic_coverage": 0.48,
          "impacted_apps": 10,
          "app_impact": 1.0
        }
      }
    },
    {
      "id": "Broker-2",
      "label": "Broker-2",
      "group": "Broker",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Routes 13 topics (52% of all topics)",
          "Impacts 10 applications (100% of all applications)"
        ],
        "metrics": {
          "topic_count": 13,
          "topic_coverage": 0.52,
          "impacted_apps": 10,
          "app_impact": 1.0
        }
      }
    },
    {
      "id": "Topic-1",
      "label": "Topic-1",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-2",
      "label": "Topic-2",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-3",
      "label": "Topic-3",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-4",
      "label": "Topic-4",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-5",
      "label": "Topic-5",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-6",
      "label": "Topic-6",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-7",
      "label": "Topic-7",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-8",
      "label": "Topic-8",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-9",
      "label": "Topic-9",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-10",
      "label": "Topic-10",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-11",
      "label": "Topic-11",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-12",
      "label": "Topic-12",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-13",
      "label": "Topic-13",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-14",
      "label": "Topic-14",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-15",
      "label": "Topic-15",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-16",
      "label": "Topic-16",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-17",
      "label": "Topic-17",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-18",
      "label": "Topic-18",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-19",
      "label": "Topic-19",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-20",
      "label": "Topic-20",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-21",
      "label": "Topic-21",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-22",
      "label": "Topic-22",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-23",
      "label": "Topic-23",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-24",
      "label": "Topic-24",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Topic-25",
      "label": "Topic-25",
      "group": "Topic",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Node-1",
      "label": "Node-1",
      "group": "Node",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Hosts 1 brokers (50% of all brokers)",
          "Hosts one or more critical brokers"
        ],
        "metrics": {
          "service_count": 4,
          "service_density": 0.3333333333333333,
          "broker_hosts": 1,
          "broker_hosting_ratio": 0.5
        }
      }
    },
    {
      "id": "Node-2",
      "label": "Node-2",
      "group": "Node",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Node-3",
      "label": "Node-3",
      "group": "Node",
      "critical": false,
      "critical_info": {}
    },
    {
      "id": "Node-4",
      "label": "Node-4",
      "group": "Node",
      "critical": true,
      "critical_info": {
        "reasons": [
          "Hosts 1 brokers (50% of all brokers)",
          "Hosts one or more critical brokers"
        ],
        "metrics": {
          "service_count": 3,
          "service_density": 0.25,
          "broker_hosts": 1,
          "broker_hosting_ratio": 0.5
        }
      }
    }
  ],
  "links": [
    {
      "source": "App-1",
      "target": "Node-1",
      "type": "RUNS_ON"
    },
    {
      "source": "App-1",
      "target": "Topic-9",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-1",
      "target": "Topic-4",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-1",
      "target": "Topic-13",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-1",
      "target": "Topic-8",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-1",
      "target": "Topic-20",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-1",
      "target": "App-4",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-1",
      "target": "App-5",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-1",
      "target": "App-6",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-1",
      "target": "App-7",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-1",
      "target": "Broker-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-1",
      "target": "Broker-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-2",
      "target": "Node-2",
      "type": "RUNS_ON"
    },
    {
      "source": "App-2",
      "target": "Topic-7",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-2",
      "target": "Topic-14",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-2",
      "target": "Topic-23",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-2",
      "target": "Topic-19",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-2",
      "target": "Topic-16",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-2",
      "target": "Topic-24",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-2",
      "target": "Topic-4",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-2",
      "target": "Topic-18",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-2",
      "target": "Topic-12",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-2",
      "target": "App-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-2",
      "target": "App-3",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-2",
      "target": "App-4",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-2",
      "target": "App-6",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-2",
      "target": "App-7",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-2",
      "target": "App-8",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-2",
      "target": "App-9",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-2",
      "target": "App-10",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-2",
      "target": "Broker-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-2",
      "target": "Broker-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-3",
      "target": "Node-3",
      "type": "RUNS_ON"
    },
    {
      "source": "App-3",
      "target": "Topic-7",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-3",
      "target": "Topic-16",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-3",
      "target": "Topic-19",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-3",
      "target": "Topic-12",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-3",
      "target": "Topic-6",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-3",
      "target": "Topic-20",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-3",
      "target": "Topic-25",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-3",
      "target": "Topic-5",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-3",
      "target": "Topic-14",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-3",
      "target": "App-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-3",
      "target": "App-4",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-3",
      "target": "App-5",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-3",
      "target": "App-7",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-3",
      "target": "App-8",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-3",
      "target": "App-9",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-3",
      "target": "App-10",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-3",
      "target": "Broker-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-3",
      "target": "Broker-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-4",
      "target": "Node-4",
      "type": "RUNS_ON"
    },
    {
      "source": "App-4",
      "target": "Topic-22",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-4",
      "target": "Topic-20",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-4",
      "target": "Topic-16",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-4",
      "target": "Topic-25",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-4",
      "target": "Topic-14",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-4",
      "target": "Topic-17",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-4",
      "target": "Topic-7",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-4",
      "target": "Topic-19",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-4",
      "target": "Topic-13",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-4",
      "target": "Topic-2",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-4",
      "target": "App-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-4",
      "target": "App-3",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-4",
      "target": "App-5",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-4",
      "target": "App-7",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-4",
      "target": "App-9",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-4",
      "target": "App-10",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-4",
      "target": "Broker-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-4",
      "target": "Broker-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-5",
      "target": "Node-1",
      "type": "RUNS_ON"
    },
    {
      "source": "App-5",
      "target": "Topic-15",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-5",
      "target": "Topic-20",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-5",
      "target": "Topic-25",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-5",
      "target": "Topic-5",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-5",
      "target": "Topic-8",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-5",
      "target": "Topic-7",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-5",
      "target": "Topic-22",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-5",
      "target": "Topic-16",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-5",
      "target": "Topic-23",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-5",
      "target": "Topic-18",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-5",
      "target": "Topic-11",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-5",
      "target": "App-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-5",
      "target": "App-3",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-5",
      "target": "App-4",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-5",
      "target": "App-6",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-5",
      "target": "App-7",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-5",
      "target": "App-8",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-5",
      "target": "App-9",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-5",
      "target": "App-10",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-5",
      "target": "Broker-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-5",
      "target": "Broker-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-6",
      "target": "Node-2",
      "type": "RUNS_ON"
    },
    {
      "source": "App-6",
      "target": "Topic-24",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-6",
      "target": "Topic-9",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-6",
      "target": "Topic-23",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-6",
      "target": "Topic-10",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-6",
      "target": "Topic-8",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-6",
      "target": "Topic-6",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-6",
      "target": "Topic-15",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-6",
      "target": "Topic-14",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-6",
      "target": "Topic-5",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-6",
      "target": "Topic-25",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-6",
      "target": "Topic-19",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-6",
      "target": "App-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-6",
      "target": "App-3",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-6",
      "target": "App-5",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-6",
      "target": "App-7",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-6",
      "target": "App-8",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-6",
      "target": "App-9",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-6",
      "target": "Broker-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-6",
      "target": "Broker-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-7",
      "target": "Node-3",
      "type": "RUNS_ON"
    },
    {
      "source": "App-7",
      "target": "Topic-15",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-7",
      "target": "Topic-21",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-7",
      "target": "Topic-23",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-7",
      "target": "Topic-13",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-7",
      "target": "Topic-5",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-7",
      "target": "Topic-12",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-7",
      "target": "Topic-24",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-7",
      "target": "App-5",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-7",
      "target": "App-6",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-7",
      "target": "App-10",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-7",
      "target": "Broker-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-7",
      "target": "Broker-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-8",
      "target": "Node-4",
      "type": "RUNS_ON"
    },
    {
      "source": "App-8",
      "target": "Topic-23",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-8",
      "target": "Topic-6",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-8",
      "target": "Topic-4",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-8",
      "target": "Topic-1",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-8",
      "target": "Topic-24",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-8",
      "target": "Topic-9",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-8",
      "target": "Topic-18",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-8",
      "target": "Topic-21",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-8",
      "target": "App-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-8",
      "target": "App-6",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-8",
      "target": "App-7",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-8",
      "target": "App-9",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-8",
      "target": "Broker-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-8",
      "target": "Broker-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-9",
      "target": "Node-1",
      "type": "RUNS_ON"
    },
    {
      "source": "App-9",
      "target": "Topic-6",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-9",
      "target": "Topic-18",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-9",
      "target": "Topic-19",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-9",
      "target": "Topic-14",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-9",
      "target": "Topic-25",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-9",
      "target": "Topic-22",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-9",
      "target": "Topic-12",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-9",
      "target": "App-4",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-9",
      "target": "App-5",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-9",
      "target": "App-10",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-9",
      "target": "Broker-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-9",
      "target": "Broker-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-10",
      "target": "Node-2",
      "type": "RUNS_ON"
    },
    {
      "source": "App-10",
      "target": "Topic-17",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-10",
      "target": "Topic-12",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-10",
      "target": "Topic-23",
      "type": "PUBLISHES_TO"
    },
    {
      "source": "App-10",
      "target": "Topic-21",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-10",
      "target": "Topic-2",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-10",
      "target": "Topic-8",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-10",
      "target": "Topic-14",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-10",
      "target": "Topic-16",
      "type": "SUBSCRIBES_TO"
    },
    {
      "source": "App-10",
      "target": "App-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-10",
      "target": "App-3",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-10",
      "target": "App-4",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-10",
      "target": "App-6",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-10",
      "target": "App-7",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-10",
      "target": "App-9",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-10",
      "target": "Broker-1",
      "type": "DEPENDS_ON"
    },
    {
      "source": "App-10",
      "target": "Broker-2",
      "type": "DEPENDS_ON"
    },
    {
      "source": "Broker-1",
      "target": "Node-4",
      "type": "RUNS_ON"
    },
    {
      "source": "Broker-1",
      "target": "Topic-1",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Topic-2",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Topic-3",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Topic-4",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Topic-5",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Topic-6",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Topic-7",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Topic-8",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Topic-9",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Topic-10",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Topic-11",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Topic-12",
      "type": "ROUTES"
    },
    {
      "source": "Broker-1",
      "target": "Broker-2",
      "type": "CONNECTS_TO"
    },
    {
      "source": "Broker-2",
      "target": "Node-1",
      "type": "RUNS_ON"
    },
    {
      "source": "Broker-2",
      "target": "Topic-13",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-14",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-15",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-16",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-17",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-18",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-19",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-20",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-21",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-22",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-23",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-24",
      "type": "ROUTES"
    },
    {
      "source": "Broker-2",
      "target": "Topic-25",
      "type": "ROUTES"
    },
    {
      "source": "Node-1",
      "target": "Node-3",
      "type": "CONNECTS_TO"
    },
    {
      "source": "Node-1",
      "target": "Node-4",
      "type": "CONNECTS_TO"
    },
    {
      "source": "Node-2",
      "target": "Node-3",
      "type": "CONNECTS_TO"
    },
    {
      "source": "Node-2",
      "target": "Node-4",
      "type": "CONNECTS_TO"
    },
    {
      "source": "Node-3",
      "target": "Node-4",
      "type": "CONNECTS_TO"
    }
  ],
  "impact": {}
};
var recommendationsData = {
  "redundancy": [
    "Implement redundant routing for topics managed by Broker-1 (12 topics, 48% of system)",
    "Implement broker clustering for Broker-1 to reduce single point of failure (impacts 10 applications)",
    "Implement redundant routing for topics managed by Broker-2 (13 topics, 52% of system)",
    "Implement broker clustering for Broker-2 to reduce single point of failure (impacts 10 applications)",
    "Separate critical brokers from node Node-1 onto dedicated infrastructure (currently hosts 1 brokers)",
    "Separate critical brokers from node Node-4 onto dedicated infrastructure (currently hosts 1 brokers)",
    "Implement redundant publishers for topics exclusively published by App-6 (3 exclusive topics)",
    "Implement redundant publishers for topics exclusively published by App-5 (3 exclusive topics)",
    "Implement redundant publishers for topics exclusively published by App-7 (3 exclusive topics)",
    "Consider increasing broker count for improved resilience (minimum 3 recommended)"
  ],
  "load_balancing": [
    "Significant broker imbalance detected. Consider redistributing topics across brokers more evenly.",
    "Multiple critical infrastructure nodes detected. Consider redistributing services more evenly.",
    "Redistribute services from overloaded node Node-1 (4 services, 33% of system)"
  ],
  "decoupling": [
    "Reduce dependencies on App-6 by implementing mediator topics (6 dependents)",
    "Reduce dependencies on App-10 by implementing mediator topics (6 dependents)",
    "Reduce dependencies on App-8 by implementing mediator topics (4 dependents)",
    "Reduce dependencies on App-9 by implementing mediator topics (7 dependents)",
    "Reduce dependencies on App-4 by implementing mediator topics (6 dependents)",
    "Reduce dependencies on App-2 by implementing mediator topics (5 dependents)",
    "Reduce dependencies on App-3 by implementing mediator topics (5 dependents)",
    "Reduce dependencies on App-5 by implementing mediator topics (6 dependents)",
    "Reduce dependencies on App-7 by implementing mediator topics (8 dependents)"
  ],
  "monitoring": []
};