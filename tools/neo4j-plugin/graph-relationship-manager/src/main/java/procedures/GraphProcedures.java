package procedures;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Result;
import org.neo4j.graphdb.Transaction;
import org.neo4j.procedure.Context;
import org.neo4j.procedure.Description;
import org.neo4j.procedure.Mode;
import org.neo4j.procedure.Name;
import org.neo4j.procedure.Procedure;

public class GraphProcedures {

    // ============================
    // RETURN CLASSES
    // ============================

    public static class BooleanResult {
        public final Boolean success;
        public BooleanResult(Boolean success) { this.success = success; }
    }

    public static class NodeRemoveResult {
        public final Boolean success;
        public final List<String> orphanedApps;
        public NodeRemoveResult(Boolean success, List<String> orphanedApps) {
            this.success = success;
            this.orphanedApps = orphanedApps;
        }
    }

    public static class AppRemoveResult {
        public final Boolean success;
        public final Boolean appDeleted;
        public AppRemoveResult(Boolean success, Boolean appDeleted) {
            this.success = success;
            this.appDeleted = appDeleted;
        }
    }

    // ============================
    // CYPHER QUERIES FROM DBMANAGER
    // ============================

    private static final String Q_MERGE_NODE = "MERGE (n:Node {id: $id}) " +
            "SET n.name = $name, n.type = 'Node' " +
            "RETURN true AS created";

    private static final String Q_CREATE_NODE_ID_ONLY = "MERGE (n:Node {id: $id}) " +
            "ON CREATE SET n.name = $id, n.type = 'Node' " +
            "RETURN true AS created";

    private static final String Q_CREATE_TOPIC_IF_NOT_EXISTS = "OPTIONAL MATCH (t:Topic {id: $id}) " +
            "WITH t WHERE t IS NULL " +
            "CREATE (new:Topic {id: $id, name: $name, type: 'Topic'}) " +
            "RETURN true AS created";

    private static final String Q_MERGE_APP = "MERGE (a:Application {name: $name}) " +
            "ON CREATE SET a.id = $name, a.type = 'Application' " +
            "RETURN a.name = $name AS created";

    private static final String Q_CREATE_RUNS_ON_IF_NOT_EXISTS = "MATCH (n:Node {id: $id}) " +
            "WITH n " +
            "MERGE (a:Application {name: $name}) " +
            "ON CREATE SET a.id = $name, a.type = 'Application' " +
            "WITH a, n " +
            "OPTIONAL MATCH (a)-[existing:RUNS_ON]->(n) " +
            "WITH a, n, existing WHERE existing IS NULL " +
            "CREATE (a)-[:RUNS_ON]->(n) " +
            "RETURN true AS created";

    private static final String Q_CREATE_SUBSCRIBES_IF_NOT_EXISTS = "MATCH (t:Topic {id: $id}) " +
            "WITH t " +
            "MERGE (a:Application {name: $name}) " +
            "ON CREATE SET a.id = $name, a.type = 'Application' " +
            "WITH a, t " +
            "OPTIONAL MATCH (a)-[existing:SUBSCRIBES_TO]->(t) " +
            "WITH a, t, existing WHERE existing IS NULL " +
            "CREATE (a)-[:SUBSCRIBES_TO]->(t) " +
            "RETURN true AS created";

    private static final String Q_CREATE_PUBLISHES_IF_NOT_EXISTS = "MATCH (t:Topic {id: $id}) " +
            "WITH t " +
            "MERGE (a:Application {name: $name}) " +
            "ON CREATE SET a.id = $name, a.type = 'Application' " +
            "WITH a, t " +
            "OPTIONAL MATCH (a)-[existing:PUBLISHES_TO]->(t) " +
            "WITH a, t, existing WHERE existing IS NULL " +
            "CREATE (a)-[:PUBLISHES_TO]->(t) " +
            "RETURN true AS created";

    private static final String Q_CONNECT_TO_ALL = "MATCH (n:Node {id: $id}) " +
            "MATCH (other:Node) WHERE other.id <> $id " +
            "MERGE (n)-[:CONNECTS_TO]->(other) " +
            "MERGE (other)-[:CONNECTS_TO]->(n)";

    private static final String Q_DELETE_RUNS_ON = "MATCH (a:Application {name: $name})-[r:RUNS_ON]->(n:Node {id: $id}) "
            + "DELETE r RETURN true AS ok";

    private static final String Q_DELETE_SUBSCRIBES = "MATCH (a:Application {name: $name})-[r:SUBSCRIBES_TO]->(t:Topic {id: $id}) "
            + "DELETE r RETURN true AS ok";

    private static final String Q_DELETE_PUBLISHES = "MATCH (a:Application {name: $name})-[r:PUBLISHES_TO]->(t:Topic {id: $id}) "
            + "DELETE r RETURN true AS ok";

    private static final String Q_DELETE_ORPHAN_APP_BY_NAME = "MATCH (a:Application {name: $name}) " +
            "WHERE NOT EXISTS { MATCH (a)-[:RUNS_ON]->(:Node) } " +
            "DETACH DELETE a RETURN true AS ok";

    private static final String Q_FIND_AND_DELETE_ORPHAN_APPS = "MATCH (a:Application)-[:RUNS_ON]->(n:Node {id: $id}) "
            + "WHERE NOT EXISTS { MATCH (a)-[:RUNS_ON]->(other:Node) WHERE other.id <> $id } " +
            "WITH a, a.name AS name " +
            "DETACH DELETE a " +
            "RETURN name";

    private static final String Q_DELETE_NODE = "MATCH (n:Node {id: $id}) DETACH DELETE n RETURN true AS ok";

    // --- BROKER OPERATIONS ---

    private static final String Q_ADD_BROKER = "MATCH (n:Node {name: $nodeName}) " +
            "MERGE (b:Broker {id: 'BROKER_' + n.id}) " +
            "ON CREATE SET b.type = 'Broker', b.name = 'BROKER', b.isPrimary = false " +
            "MERGE (b)-[:RUNS_ON]->(n) " +
            "RETURN b.id as brokerId";

    private static final String Q_PROMOTE_BROKER = "MATCH (b:Broker {id: $brokerId}) SET b.isPrimary = true";

    // Alternate handover without APOC for safety if APOC not strictly present:
    private static final String Q_HANDOVER_BROKERS = "MATCH (oldPrimary:Broker {isPrimary: true}) " +
            "WHERE oldPrimary.id <> $newBrokerId " +
            "SET oldPrimary.isPrimary = false " +
            "WITH oldPrimary " +
            "MATCH (newPrimary:Broker {id: $newBrokerId}) " +
            "OPTIONAL MATCH (oldPrimary)-[r:ROUTES]->(t:Topic) " +
            "FOREACH (ignoreMe IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END | " +
            "  CREATE (newPrimary)-[:ROUTES]->(t) " +
            "  DELETE r " +
            ") " +
            "RETURN oldPrimary.id as oldBrokerId";

    private static final String Q_CHECK_AND_DELETE_USELESS_BROKERS = "MATCH (b:Broker) " +
            "WHERE NOT EXISTS { MATCH (b)-[:RUNS_ON]->(:Node) } " +
            "AND NOT EXISTS { MATCH (b)-[:ROUTES]->(:Topic) } " +
            "DETACH DELETE b";

    private static final String Q_CLEANUP_ORPHAN_TOPICS = "MATCH (t:Topic) " +
            "WHERE NOT EXISTS { MATCH (:Broker)-[:ROUTES]->(t) } " +
            "DETACH DELETE t";

    private static final String Q_ROUTE_ALL_ORPHAN_TOPICS_TO_BROKER = "MATCH (b:Broker {id: $brokerId}), (t:Topic) " +
            "WHERE NOT EXISTS { MATCH (:Broker)-[:ROUTES]->(t) } " +
            "CREATE (b)-[:ROUTES]->(t)";

    private static final String Q_DELETE_TOPIC = "MATCH (t:Topic {id: $id}) DETACH DELETE t RETURN true AS ok";

    private static final String Q_DELETE_APP = "MATCH (a:Application {name: $name}) DETACH DELETE a RETURN true AS ok";

    private static final String Q_FIND_APP = "MATCH (a:Application {name: $name}) RETURN a LIMIT 1";

    private static final String Q_RENAME_APP = "MATCH (a:Application {name: $oldName}) SET a.id = $newName, a.name = $newName";

    private static final String[] REL_TYPES = { "RUNS_ON", "SUBSCRIBES_TO", "PUBLISHES_TO" };
    private static final String[] Q_TRANSFER_RELS = new String[REL_TYPES.length];
    static {
        String fmt = "MATCH (p:Application {name: $pName})-[r:%s]->(target) " +
                "MATCH (real:Application {name: $rName}) " +
                "WHERE NOT EXISTS { MATCH (real)-[:%s]->(target) } " +
                "CREATE (real)-[:%s]->(target)";
        for (int i = 0; i < REL_TYPES.length; i++) {
            Q_TRANSFER_RELS[i] = String.format(fmt, REL_TYPES[i], REL_TYPES[i], REL_TYPES[i]);
        }
    }

    private static final String Q_ADD_ALIAS = "MATCH (real:Application {name: $rName}) " +
            "SET real.aliases = CASE " +
            "  WHEN real.aliases IS NULL THEN [$pName] " +
            "  WHEN NOT $pName IN real.aliases THEN real.aliases + $pName " +
            "  ELSE real.aliases " +
            "END";

    // ============================
    // CYPHER QUERIES FROM DerivedRelationsProcedures
    // ============================

    private static final String Q_INC_UPDATE_APP_DEPS_ON_SUB = "MATCH (subscriber:Application {name: $appName}) " +
        "MATCH (t:Topic {id: $topicId}) " +
        "MATCH (t)<-[:PUBLISHES_TO]-(publisher:Application) " +
        "WHERE publisher.name <> subscriber.name " +
        "MERGE (subscriber)-[:DEPENDS_ON {dependency_type: 'app_to_app'}]->(publisher)";

    private static final String Q_INC_UPDATE_APP_DEPS_ON_PUB = "MATCH (publisher:Application {name: $appName}) " +
        "MATCH (t:Topic {id: $topicId}) " +
        "MATCH (t)<-[:SUBSCRIBES_TO]-(subscriber:Application) " +
        "WHERE subscriber.name <> publisher.name " +
        "MERGE (subscriber)-[:DEPENDS_ON {dependency_type: 'app_to_app'}]->(publisher)";

    private static final String Q_INC_UPDATE_NODE_DEPS = "MATCH (app:Application {name: $appName})-[:RUNS_ON]->(thisNode:Node {id: $nodeId}) " +
        "WITH app, thisNode " +
        "OPTIONAL MATCH (app)-[:DEPENDS_ON]->(targetApp:Application)-[:RUNS_ON]->(targetNode:Node) " +
        "WHERE targetNode <> thisNode " +
        "FOREACH (ignoreMe IN CASE WHEN targetNode IS NOT NULL THEN [1] ELSE [] END | " +
        "  MERGE (thisNode)-[:DEPENDS_ON {dependency_type: 'node_to_node'}]->(targetNode) " +
        ") " +
        "WITH app, thisNode " +
        "OPTIONAL MATCH (sourceApp:Application)-[:DEPENDS_ON]->(app)-[:RUNS_ON]->(sourceNode:Node) " +
        "WHERE sourceNode <> thisNode " +
        "FOREACH (ignoreMe IN CASE WHEN sourceNode IS NOT NULL THEN [1] ELSE [] END | " +
        "  MERGE (sourceNode)-[:DEPENDS_ON {dependency_type: 'node_to_node'}]->(thisNode) " +
        ")";

    private static final String Q_INC_CLEANUP_APP_DEPS = "MATCH (a:Application {name: $appName})-[d:DEPENDS_ON]-() " +
        "WHERE d.dependency_type = 'app_to_app' " +
        "WITH d, startNode(d) as sub, endNode(d) as pub " +
        "WHERE NOT EXISTS { MATCH (sub)-[:SUBSCRIBES_TO]->(:Topic)<-[:PUBLISHES_TO]-(pub) } " +
        "DELETE d";

    private static final String Q_INC_CLEANUP_NODE_DEPS = "MATCH (n:Node {id: $nodeId})-[d:DEPENDS_ON]-() " +
        "WHERE d.dependency_type = 'node_to_node' " +
        "WITH d, startNode(d) as n1, endNode(d) as n2 " +
        "WHERE NOT EXISTS { MATCH (n1)<-[:RUNS_ON]-(:Application)-[:DEPENDS_ON]->(:Application)-[:RUNS_ON]->(n2) } " +
        "DELETE d";

    @Context
    public GraphDatabaseService db;

    // ============================
    // NEW PROCEDURES (MIGRATED)
    // ============================

    @Procedure(name = "custom.addNodeIdOnly", mode = Mode.WRITE)
    @Description("CALL custom.addNodeIdOnly(id) YIELD success")
    public Stream<BooleanResult> addNodeIdOnly(@Name("id") String id) {
        try (Transaction tx = db.beginTx()) {
            boolean success = tx.execute(Q_CREATE_NODE_ID_ONLY, Map.of("id", id)).hasNext();
            if (success) {
                tx.execute(Q_CONNECT_TO_ALL, Map.of("id", id));
            }
            tx.commit();
            return Stream.of(new BooleanResult(success));
        }
    }

    @Procedure(name = "custom.addNode", mode = Mode.WRITE)
    @Description("CALL custom.addNode(id, name) YIELD success")
    public Stream<BooleanResult> addNode(@Name("id") String id, @Name("name") String name) {
        try (Transaction tx = db.beginTx()) {
            boolean success = tx.execute(Q_MERGE_NODE, Map.of("id", id, "name", name)).hasNext();
            if (success) {
                tx.execute(Q_CONNECT_TO_ALL, Map.of("id", id));
            }
            tx.commit();
            return Stream.of(new BooleanResult(success));
        }
    }

    @Procedure(name = "custom.addTopic", mode = Mode.WRITE)
    @Description("CALL custom.addTopic(id, name) YIELD success")
    public Stream<BooleanResult> addTopic(@Name("id") String id, @Name("name") String name) {
        try (Transaction tx = db.beginTx()) {
            // Reject topic creation if no ALIVE Primary Broker exists (must have RUNS_ON to a Node; zombie brokers don't count)
            boolean hasAlivePrimary = tx.execute("MATCH (b:Broker {isPrimary: true})-[:RUNS_ON]->(:Node) RETURN b LIMIT 1").hasNext();
            if (!hasAlivePrimary) {
                tx.commit();
                return Stream.of(new BooleanResult(false));
            }
            boolean success = tx.execute(Q_CREATE_TOPIC_IF_NOT_EXISTS, Map.of("id", id, "name", name)).hasNext();
            if (success) {
                tx.execute("MATCH (b:Broker {isPrimary: true})-[:RUNS_ON]->(:Node), (t:Topic {id: $id}) MERGE (b)-[:ROUTES]->(t)", Map.of("id", id));
            }
            tx.commit();
            return Stream.of(new BooleanResult(success));
        }
    }

    @Procedure(name = "custom.addApplication", mode = Mode.WRITE)
    @Description("CALL custom.addApplication(prefixedId, sanitizedName) YIELD success")
    public Stream<BooleanResult> addApplication(@Name("prefixedId") String prefixedId, @Name("sanitizedName") String sanitizedName) {
        try (Transaction tx = db.beginTx()) {
            boolean placeholderExists = tx.execute(Q_FIND_APP, Map.of("name", prefixedId)).hasNext();
            boolean realExists = tx.execute(Q_FIND_APP, Map.of("name", sanitizedName)).hasNext();

            Map<String, Object> transferParams = Map.of("pName", prefixedId, "rName", sanitizedName);
            if (placeholderExists && realExists) {
                for (String q : Q_TRANSFER_RELS) {
                    tx.execute(q, transferParams);
                }
                tx.execute(Q_ADD_ALIAS, transferParams);
                tx.execute(Q_DELETE_APP, Map.of("name", prefixedId));
                tx.commit();
                return Stream.of(new BooleanResult(true));
            } else if (placeholderExists) {
                tx.execute(Q_RENAME_APP, Map.of("oldName", prefixedId, "newName", sanitizedName));
                tx.commit();
                return Stream.of(new BooleanResult(true));
            } else if (!realExists) {
                tx.execute(Q_MERGE_APP, Map.of("name", sanitizedName));
                tx.commit();
                return Stream.of(new BooleanResult(true));
            }
            tx.commit();
            return Stream.of(new BooleanResult(false));
        }
    }

    @Procedure(name = "custom.removeNode", mode = Mode.WRITE)
    @Description("CALL custom.removeNode(id) YIELD success, orphanedApps")
    public Stream<NodeRemoveResult> removeNode(@Name("id") String id) {
        try (Transaction tx = db.beginTx()) {
            Result orphans = tx.execute(Q_FIND_AND_DELETE_ORPHAN_APPS, Map.of("id", id));
            List<String> orphanedApps = new ArrayList<>();
            while (orphans.hasNext()) {
                orphanedApps.add((String) orphans.next().get("name"));
            }
            boolean success = tx.execute(Q_DELETE_NODE, Map.of("id", id)).hasNext();
            
            // Broker cleanup: delete brokers with no Node AND no Routes.
            // A broker that still has ROUTES stays as a zombie until a new primary takes over.
            tx.execute(Q_CHECK_AND_DELETE_USELESS_BROKERS);

            // Safety net: if no broker exists at all, orphan topics are cleaned up.
            tx.execute(Q_CLEANUP_ORPHAN_TOPICS);
            
            tx.commit();
            return Stream.of(new NodeRemoveResult(success, orphanedApps));
        }
    }

    /**
     * Integrates Broker logic based on Worker ping.
     * All brokers are initially created as non-primary (isPrimary=false).
     * If isPrimary is asserted true, the broker is promoted: the pre-existing Primary Broker
     * is demoted, its ROUTES are transferred to the new primary, and orphan topics are re-routed.
     * Useless brokers (no RUNS_ON, no ROUTES) are garbage-collected at the end.
     */
    @Procedure(name = "custom.addBroker", mode = Mode.WRITE)
    @Description("CALL custom.addBroker(nodeName, isPrimary) YIELD success")
    public Stream<BooleanResult> addBroker(@Name("nodeName") String nodeName, @Name("isPrimary") Boolean isPrimary) {
        try (Transaction tx = db.beginTx()) {
            // 1. Always create broker as non-primary first
            Result r = tx.execute(Q_ADD_BROKER, Map.of("nodeName", nodeName));
            if (!r.hasNext()) {
                tx.commit();
                return Stream.of(new BooleanResult(false));
            }
            String newBrokerId = (String) r.next().get("brokerId");

            // 2. If isPrimary requested, promote this broker
            if (isPrimary != null && isPrimary) {
                // 2a. Handover: demote old primary and transfer its routes to this broker
                tx.execute(Q_HANDOVER_BROKERS, Map.of("newBrokerId", newBrokerId));

                // 2b. Promote this broker to primary
                tx.execute(Q_PROMOTE_BROKER, Map.of("brokerId", newBrokerId));

                // 2c. Route any orphan topics (not routed to any broker) to the new primary
                tx.execute(Q_ROUTE_ALL_ORPHAN_TOPICS_TO_BROKER, Map.of("brokerId", newBrokerId));

                // 2d. Cleanup any brokers that were stripped of their roles and have no node
                tx.execute(Q_CHECK_AND_DELETE_USELESS_BROKERS);
            }

            tx.commit();
            return Stream.of(new BooleanResult(true));
        }
    }

    @Procedure(name = "custom.removeTopic", mode = Mode.WRITE)
    @Description("CALL custom.removeTopic(id) YIELD success")
    public Stream<BooleanResult> removeTopic(@Name("id") String id) {
        try (Transaction tx = db.beginTx()) {
            boolean success = tx.execute(Q_DELETE_TOPIC, Map.of("id", id)).hasNext();
            tx.commit();
            return Stream.of(new BooleanResult(success));
        }
    }

    @Procedure(name = "custom.removeApplication", mode = Mode.WRITE)
    @Description("CALL custom.removeApplication(appName) YIELD success")
    public Stream<BooleanResult> removeApplication(@Name("appName") String appName) {
        try (Transaction tx = db.beginTx()) {
            boolean success = tx.execute(Q_DELETE_APP, Map.of("name", appName)).hasNext();
            tx.commit();
            return Stream.of(new BooleanResult(success));
        }
    }

    @Procedure(name = "custom.addRunsOn", mode = Mode.WRITE)
    @Description("CALL custom.addRunsOn(appName, nodeId) YIELD success")
    public Stream<BooleanResult> addRunsOn(@Name("appName") String appName, @Name("nodeId") String nodeId) {
        try (Transaction tx = db.beginTx()) {
            boolean success = tx.execute(Q_CREATE_RUNS_ON_IF_NOT_EXISTS, Map.of("name", appName, "id", nodeId)).hasNext();
            if (success) {
                tx.execute(Q_INC_UPDATE_NODE_DEPS, Map.of("appName", appName, "nodeId", nodeId));
            }
            tx.commit();
            return Stream.of(new BooleanResult(success));
        }
    }

    @Procedure(name = "custom.removeRunsOn", mode = Mode.WRITE)
    @Description("CALL custom.removeRunsOn(appName, nodeId) YIELD success, appDeleted")
    public Stream<AppRemoveResult> removeRunsOn(@Name("appName") String appName, @Name("nodeId") String nodeId) {
        try (Transaction tx = db.beginTx()) {
            boolean success = tx.execute(Q_DELETE_RUNS_ON, Map.of("name", appName, "id", nodeId)).hasNext();
            if (!success) {
                tx.commit();
                return Stream.of(new AppRemoveResult(false, false));
            }
            tx.execute(Q_DELETE_ORPHAN_APP_BY_NAME, Map.of("name", appName));
            tx.execute(Q_INC_CLEANUP_NODE_DEPS, Map.of("nodeId", nodeId));
            
            boolean appDeleted = !tx.execute(Q_FIND_APP, Map.of("name", appName)).hasNext();
            tx.commit();
            return Stream.of(new AppRemoveResult(true, appDeleted));
        }
    }

    private void triggerNodeDepUpdatesForApp(Transaction tx, String appName) {
        Result nodes = tx.execute("MATCH (a:Application {name: $appName})-[:RUNS_ON]->(n) RETURN n.id as id", Map.of("appName", appName));
        while (nodes.hasNext()) {
            String nid = (String) nodes.next().get("id");
            tx.execute(Q_INC_UPDATE_NODE_DEPS, Map.of("appName", appName, "nodeId", nid));
        }
    }

    @Procedure(name = "custom.addSubscribesTo", mode = Mode.WRITE)
    @Description("CALL custom.addSubscribesTo(appName, topicId) YIELD success")
    public Stream<BooleanResult> addSubscribesTo(@Name("appName") String appName, @Name("topicId") String topicId) {
        try (Transaction tx = db.beginTx()) {
            boolean success = tx.execute(Q_CREATE_SUBSCRIBES_IF_NOT_EXISTS, Map.of("name", appName, "id", topicId)).hasNext();
            if (success) {
                tx.execute(Q_INC_UPDATE_APP_DEPS_ON_SUB, Map.of("appName", appName, "topicId", topicId));
                triggerNodeDepUpdatesForApp(tx, appName);
            }
            tx.commit();
            return Stream.of(new BooleanResult(success));
        }
    }

    @Procedure(name = "custom.removeSubscribesTo", mode = Mode.WRITE)
    @Description("CALL custom.removeSubscribesTo(appName, topicId) YIELD success")
    public Stream<BooleanResult> removeSubscribesTo(@Name("appName") String appName, @Name("topicId") String topicId) {
        try (Transaction tx = db.beginTx()) {
            boolean success = tx.execute(Q_DELETE_SUBSCRIBES, Map.of("name", appName, "id", topicId)).hasNext();
            if (success) {
                tx.execute(Q_INC_CLEANUP_APP_DEPS, Map.of("appName", appName));
                triggerNodeDepUpdatesForApp(tx, appName);
            }
            tx.commit();
            return Stream.of(new BooleanResult(success));
        }
    }

    @Procedure(name = "custom.addPublishesTo", mode = Mode.WRITE)
    @Description("CALL custom.addPublishesTo(appName, topicId) YIELD success")
    public Stream<BooleanResult> addPublishesTo(@Name("appName") String appName, @Name("topicId") String topicId) {
        try (Transaction tx = db.beginTx()) {
            boolean success = tx.execute(Q_CREATE_PUBLISHES_IF_NOT_EXISTS, Map.of("name", appName, "id", topicId)).hasNext();
            if (success) {
                tx.execute(Q_INC_UPDATE_APP_DEPS_ON_PUB, Map.of("appName", appName, "topicId", topicId));
                triggerNodeDepUpdatesForApp(tx, appName);
            }
            tx.commit();
            return Stream.of(new BooleanResult(success));
        }
    }

    @Procedure(name = "custom.removePublishesTo", mode = Mode.WRITE)
    @Description("CALL custom.removePublishesTo(appName, topicId) YIELD success")
    public Stream<BooleanResult> removePublishesTo(@Name("appName") String appName, @Name("topicId") String topicId) {
        try (Transaction tx = db.beginTx()) {
            boolean success = tx.execute(Q_DELETE_PUBLISHES, Map.of("name", appName, "id", topicId)).hasNext();
            if (success) {
                tx.execute(Q_INC_CLEANUP_APP_DEPS, Map.of("appName", appName));
                triggerNodeDepUpdatesForApp(tx, appName);
            }
            tx.commit();
            return Stream.of(new BooleanResult(success));
        }
    }

    // ============================
    // OLD PROCEDURES (PRESERVED)
    // ============================
    
    @Procedure(name = "custom.updateAppDepsOnSub", mode = Mode.WRITE)
    @Description("CALL custom.updateAppDepsOnSub(appName, topicId)")
    public void updateAppDepsOnSub(@Name("appName") String appName, @Name("topicId") String topicId) {
        try (Transaction tx = db.beginTx()) {
            tx.execute(Q_INC_UPDATE_APP_DEPS_ON_SUB, Map.of("appName", appName, "topicId", topicId));
            tx.commit();
        }
    }

    @Procedure(name = "custom.updateAppDepsOnPub", mode = Mode.WRITE)
    @Description("CALL custom.updateAppDepsOnPub(appName, topicId)")
    public void updateAppDepsOnPub(@Name("appName") String appName, @Name("topicId") String topicId) {
        try (Transaction tx = db.beginTx()) {
            tx.execute(Q_INC_UPDATE_APP_DEPS_ON_PUB, Map.of("appName", appName, "topicId", topicId));
            tx.commit();
        }
    }

    @Procedure(name = "custom.updateNodeDeps", mode = Mode.WRITE)
    @Description("CALL custom.updateNodeDeps(appName, nodeId)")
    public void updateNodeDeps(@Name("appName") String appName, @Name("nodeId") String nodeId) {
        try (Transaction tx = db.beginTx()) {
            tx.execute(Q_INC_UPDATE_NODE_DEPS, Map.of("appName", appName, "nodeId", nodeId));
            tx.commit();
        }
    }

    @Procedure(name = "custom.cleanupAppDeps", mode = Mode.WRITE)
    @Description("CALL custom.cleanupAppDeps(appName)")
    public void cleanupAppDeps(@Name("appName") String appName) {
        try (Transaction tx = db.beginTx()) {
            tx.execute(Q_INC_CLEANUP_APP_DEPS, Map.of("appName", appName));
            tx.commit();
        }
    }

    @Procedure(name = "custom.cleanupNodeDeps", mode = Mode.WRITE)
    @Description("CALL custom.cleanupNodeDeps(nodeId)")
    public void cleanupNodeDeps(@Name("nodeId") String nodeId) {
        try (Transaction tx = db.beginTx()) {
            tx.execute(Q_INC_CLEANUP_NODE_DEPS, Map.of("nodeId", nodeId));
            tx.commit();
        }
    }
}