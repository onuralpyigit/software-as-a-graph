# Software-as-a-Graph (SaaG): Graph Relationship Manager Module

This repository is the core plugin bundle containing the **Graph Relationship Manager** written specifically for the Neo4j database as part of the **Dynamic Dataset Processing** subsystem of the Monitor DB "Software-as-a-Graph" (SaaG) architecture.

It enables graph-based applications to establish event-driven architectures, maintain Broker routings, and achieve a Self-Healing Graph by dynamically building and cleaning up "Derived" relationships across Node, App, and Topic integrations under edge-case scenarios.

## Core Features

Compared to standard Neo4j CYPHER commands, this module achieves the following:
1. **Relational Integrity:** Cleans up dangling dependencies and sweeps orphaned Topics when a Node, Topic, or App is deleted from the database.
2. **App Handover & Merging:** Safely migrates relationships from an old application entry to a newly named application entry (`newApp`), ensuring zero data loss during naming/identity shifts.
3. **Broker Routing (High Availability):** Maintains an active Primary Broker for uninterrupted Topic messaging. If a machine goes down, connection Routers attached to the falling Broker are instantly handed over to the newly promoted Primary Broker.
4. **Derived Dependencies:** If Application A publishes to a Topic that Application B is subscribed to, a `[A] -DEPENDS_ON-> [B]` graph dependency is instantly derived, and proactively deleted if the messaging bridge breaks.

## Provided Procedure Interfaces

The entire system is exposed under the `custom.*` namespace.

### 1- Machine and Server (Node) Operations
Registers physical/logical server instances into the system.
- `custom.addNodeIdOnly(id)`: Adds temporary anonymous Nodes.
- `custom.addNode(id, name)`: Creates a fully named device asset.
- `custom.removeNode(id)`: **[CRITICAL EDGE CASE]** Eliminates the target Node from the graph. If the applications running (`RUNS_ON`) on this machine are **not tied to any other device**, they are considered completely orphaned and are permanently **deleted** from the database. The weight values of `DEPENDS_ON` dependencies attached to the Node will trigger a recalculation workflow.

### 2- Communication Backbone (Broker & Topic)
Manages the messaging spine.
- `custom.addTopic(id, name)`: Defines a new messaging channel. Requires an **alive Primary Broker** (one that has `RUNS_ON` to a live Node). If no alive Primary Broker exists — including during broker handover — topic creation is **rejected** (`success=false`). Zombie brokers (primary brokers whose host node died) do not qualify.
- `custom.addBroker(nodeName, isPrimary)`: Constructs a Broker entity directly borrowing its host machine's identity in the `BROKER_<nodeid>` format, constantly named `BROKER`. Brokers are always created as non-primary (`isPrimary=false`) first.
  - **[EDGE CASE - Broker Handover]:** If the `isPrimary` parameter asserts `true`, the broker is promoted to Primary. The preexisting Primary Broker in the database is automatically demoted (`false`). All Topic Routers belonging to the old Broker are instantly migrated to the new Primary Broker safely. Any orphan Topics (not routed to any Broker) are automatically bound to the new Primary via `Q_ROUTE_ALL_ORPHAN_TOPICS_TO_BROKER`.
- `custom.removeTopic(id)`: Deletes the target Topic.

### 3- Application (App) Task and Dependencies
Describes the node an application relies on and the channels it monitors.
- `custom.addApp(name)`: Creates an empty application object devoid of any services or hosts.
- `custom.addRunsOn(appName, nodeId)`: Binds an application to a specific Node. This action incrementally increases the corresponding `node_to_node` machine dependency weights.
- `custom.removeRunsOn(appName, nodeId)`: Detaches an application from a specific server.
  - **[EDGE CASE]:** When an application is detached, if it is not bound (`RUNS_ON`) to any other machine in the Graph, the system flags it as an orphan and safely enforces a **permanent deletion (appDeleted=true)** to maintain graph integrity.
- `custom.addSubscriber(appName, topicId)` & `custom.addPublisher(appName, topicId)`: Establishes a listener or publisher bond. Triggers the subsequent creation of an `app_to_app` derived relation (`DEPENDS_ON`).
- `custom.removeSubscriber(appName, topicId)` & `custom.removePublisher(appName, topicId)`: Terminates the listening/publishing bond. Any disconnection that orphans a Topic or a dependency will force the corresponding `DEPENDS_ON` edge to be deleted from the Graph (`Q_INC_CLEANUP_APP_DEPS`).

### 4- Merge (Alias / App Consolidation)
- `custom.updateApp(currentName, newName)`: During an application rename procedure, all cumulative `RUNS_ON`, `PUBLISHES_TO`, and `SUBSCRIBES_TO` relationships from the old identity are relocated directly to the new application node. The old name is archived into the application's `aliases` property, and the loose placeholder record is deleted. This transition preserves logical linkages during blue-green deployment mappings in High-Availability clusters.

## Edge Cases Summary and Self-Healing Cycle

The plugin executes background **Garbage Collector** routines at the end of each Write Transaction to preserve integrity:
1. When a Node dies or gets removed (`custom.removeNode` fires), the Graph enforces a structural Cascade Delete severing tied arrays.
2. A Primary Broker whose host Node dies becomes a **Zombie Broker**: it retains its `ROUTES` to Topics but loses its `RUNS_ON`. It stays alive until a new Primary Broker is registered and takes over its routes via handover.
3. Once a Broker has no `RUNS_ON` and no `ROUTES` left, `Q_CHECK_AND_DELETE_USELESS_BROKERS` garbage-collects it.
4. After broker cleanup in `removeNode`, `Q_CLEANUP_ORPHAN_TOPICS` acts as a safety net: any Topics that have lost all Broker routes (i.e., no Broker exists in the system at all) are permanently deleted.
5. The moment a new Primary Broker is registered (`custom.addBroker` with `isPrimary=true`), `Q_ROUTE_ALL_ORPHAN_TOPICS_TO_BROKER` scans for any orphan Topics and binds them to the new Primary, resuscitating data flow.
6. Derived Application dependencies are solely sustained if both entities maintain an active Subscribe/Publish agreement on the mutual Topic constraint. The split second a party unsubscribes, the Dependency Arrow is dynamically re-calculated and irrecoverably vaporized from the Graph.
