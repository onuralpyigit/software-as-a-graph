package procedures;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.Test;
import org.neo4j.driver.AuthTokens;
import org.neo4j.driver.Driver;
import org.neo4j.driver.GraphDatabase;
import org.neo4j.driver.Session;

public class E2ETest {
    @Test
    public void executeComprehensiveE2E() {
        System.out.println("Starting Comprehensive E2E Tests (DBManager-free)...");
        try (Driver driver = GraphDatabase.driver("bolt://127.0.0.1:7687", AuthTokens.basic("neo4j", "password"), org.neo4j.driver.Config.builder().withoutEncryption().build());
             Session session = driver.session()) {
             
            // Setup
            session.run("MATCH (n) DETACH DELETE n");
            
            // --- SCENARIO 1: Basic Node, App, Topic, and Pub/Sub ---
            System.out.println("\n--- SCENARIO 1: Basic Pub/Sub & Derived Relationships ---");
            run(session, "CALL custom.addNode('NODE_server-1', 'Web Server 1')");
            
            // A Topic requires a Primary Broker to route to. Otherwise it is orphaned and deleted.
            run(session, "CALL custom.addNode('NODE_broker-node', 'Message Broker Node')");
            run(session, "CALL custom.addBroker('Message Broker Node', true)");

            run(session, "CALL custom.addTopic('TOPIC_events-bus', 'Main Event Bus')");
            
            run(session, "CALL custom.addApplication('APP_svc-auth', 'AuthService')");
            run(session, "CALL custom.addRunsOn('AuthService', 'NODE_server-1')");
            
            run(session, "CALL custom.addApplication('APP_svc-billing', 'BillingService')");
            run(session, "CALL custom.addRunsOn('BillingService', 'NODE_server-1')");
            
            run(session, "CALL custom.addPublishesTo('AuthService', 'TOPIC_events-bus')");
            run(session, "CALL custom.addSubscribesTo('BillingService', 'TOPIC_events-bus')");

            Thread.sleep(500); // give neo4j a tiny moment

            long appNodes = session.run("MATCH (a:Application {name: 'AuthService'}) RETURN count(a) AS c").next().get("c").asLong();
            assertTrue(appNodes == 1, "[SCENARIO 1] AuthService node created");
            
            long rels = session.run("MATCH (:Application {name: 'AuthService'})-[:RUNS_ON]->(:Node {id: 'NODE_server-1'}) RETURN count(*) AS c").next().get("c").asLong();
            assertTrue(rels == 1, "[SCENARIO 1] AuthService RUNS_ON server-1");

            long deps = session.run("MATCH (a:Application {name: 'BillingService'})-[:DEPENDS_ON {dependency_type:'app_to_app'}]->(b:Application {name: 'AuthService'}) RETURN count(*) AS c").next().get("c").asLong();
            assertTrue(deps == 1, "[SCENARIO 1] BillingService DEPENDS_ON AuthService (Derived App Relation)");
            
            // --- SCENARIO 2: Orphan App Deletion on Node Removal ---
            System.out.println("\n--- SCENARIO 2: Orphan App Deletion on Node Removal ---");
            run(session, "CALL custom.addNode('NODE_server-orphan', 'Orphan Server')");
            run(session, "CALL custom.addApplication('APP_svc-orphan', 'OrphanApp')");
            run(session, "CALL custom.addRunsOn('OrphanApp', 'NODE_server-orphan')");
            
            long c = session.run("MATCH (a:Application {name: 'OrphanApp'}) RETURN count(a) AS c").next().get("c").asLong();
            assertTrue(c == 1, "[SCENARIO 2] OrphanApp created");

            // Remove the node. The app 'OrphanApp' has no other node, so it should be deleted.
            run(session, "CALL custom.removeNode('NODE_server-orphan')");
            
            long nodeCount = session.run("MATCH (n:Node {id: 'NODE_server-orphan'}) RETURN count(n) AS c").next().get("c").asLong();
            assertTrue(nodeCount == 0, "[SCENARIO 2] server-orphan effectively deleted from DB");

            long appCount = session.run("MATCH (a:Application {name: 'OrphanApp'}) RETURN count(a) AS c").next().get("c").asLong();
            assertTrue(appCount == 0, "[SCENARIO 2] OrphanApp auto-deleted because it became an orphan");

            // --- SCENARIO 3: Non-Orphan App Retention (Runs on Multiple Nodes) ---
            System.out.println("\n--- SCENARIO 3: Non-Orphan App Retention (Runs on Multiple Nodes) ---");
            run(session, "CALL custom.addNode('NODE_server-multi-1', 'Multi Server 1')");
            run(session, "CALL custom.addNode('NODE_server-multi-2', 'Multi Server 2')");
            run(session, "CALL custom.addApplication('APP_svc-resilient', 'ResilientApp')");
            run(session, "CALL custom.addRunsOn('ResilientApp', 'NODE_server-multi-1')");
            run(session, "CALL custom.addRunsOn('ResilientApp', 'NODE_server-multi-2')");

            // Remove ONE node. The app should NOT be deleted.
            run(session, "CALL custom.removeNode('NODE_server-multi-1')");

            appCount = session.run("MATCH (a:Application {name: 'ResilientApp'}) RETURN count(a) AS c").next().get("c").asLong();
            assertTrue(appCount == 1, "[SCENARIO 3] ResilientApp NOT deleted because it still runs on server-multi-2");

            // --- SCENARIO 4: Application Merging / Alias Resolving ---
            System.out.println("\n--- SCENARIO 4: Application Merging / Alias Resolving ---");
            session.run("CREATE (a:Application {name: 'APP_1234', type:'Application'})");
            session.run("CREATE (a:Application {name: 'RealApp', type:'Application'})");
            session.run("MATCH (a:Application {name: 'APP_1234'}), (n:Node {id: 'NODE_server-1'}) CREATE (a)-[:RUNS_ON]->(n)");

            // Calling addApplication maps ID to Name and triggers merging
            run(session, "CALL custom.addApplication('APP_1234', 'RealApp')"); // PId='APP_1234', SName='RealApp'
            
            long placeholderCount = session.run("MATCH (a:Application {name: 'APP_1234'}) RETURN count(a) AS c").next().get("c").asLong();
            assertTrue(placeholderCount == 0, "[SCENARIO 4] Placeholder App APP_1234 was deleted after merge");

            long realAppRunsOn = session.run("MATCH (a:Application {name: 'RealApp'})-[:RUNS_ON]->(:Node {id: 'NODE_server-1'}) RETURN count(*) AS c").next().get("c").asLong();
            assertTrue(realAppRunsOn == 1, "[SCENARIO 4] RealApp took over relations from placeholder APP_1234");
            
            org.neo4j.driver.Record rec = session.run("MATCH (a:Application {name: 'RealApp'}) RETURN a.aliases as aliases").next();
            List<Object> aliases = rec.get("aliases").asList();
            assertTrue(aliases.contains("APP_1234"), "[SCENARIO 4] RealApp acquired the alias APP_1234");

            // --- SCENARIO 5: Relationship Removal Orchestration ---
            System.out.println("\n--- SCENARIO 5: Relationship Removal Orchestration ---");
            run(session, "CALL custom.removeSubscribesTo('BillingService', 'TOPIC_events-bus')");
            Thread.sleep(500);

            deps = session.run("MATCH (a:Application {name: 'BillingService'})-[:DEPENDS_ON]->(b:Application {name: 'AuthService'}) RETURN count(*) AS c").next().get("c").asLong();
            assertTrue(deps == 0, "[SCENARIO 5] Dependency deleted securely when SUBSCRIBES_TO was removed");

            // --- SCENARIO 6: Broker Route Handover and Lifecycle ---
            System.out.println("\n--- SCENARIO 6: Broker Route Handover and Lifecycle ---");
            run(session, "CALL custom.addNode('NODE_broker-node-1', 'Broker Server 1')");
            run(session, "CALL custom.addBroker('Broker Server 1', true)");

            run(session, "CALL custom.addTopic('TOPIC_infra-events', 'Infrastructure Events')");
            run(session, "CALL custom.addTopic('TOPIC_payment-events', 'Payment Events')");

            long primaryBrokers = session.run("MATCH (b:Broker {isPrimary: true}) RETURN count(b) AS c").next().get("c").asLong();
            assertTrue(primaryBrokers == 1, "[SCENARIO 6] Exactly 1 primary broker exists");
            
            long routes = session.run("MATCH (:Broker {isPrimary: true})-[:ROUTES]->(t:Topic) RETURN count(t) AS c").next().get("c").asLong();
            assertTrue(routes == 3, "[SCENARIO 6] All topics correctly routed to Primary Broker 1");

            // Simulate Broker 1 dying (node removed), wait 1 sec 
            run(session, "CALL custom.removeNode('NODE_broker-node-1')");
            Thread.sleep(1000);

            long bCount = session.run("MATCH (b:Broker {id: 'BROKER_NODE_broker-node-1'}) RETURN count(b) AS c").next().get("c").asLong();
            assertTrue(bCount == 1, "[SCENARIO 6] Broker 1 still exists in DB as an Orphan (No RUNS_ON)");
            
            long rCount = session.run("MATCH (a:Broker {id: 'BROKER_NODE_broker-node-1'})-[:ROUTES]->(:Topic) RETURN count(*) AS c").next().get("c").asLong();
            assertTrue(rCount == 3, "[SCENARIO 6] Orphaned Broker 1 still holds the Routes");

            // Simulate Worker activating 5 seconds later and assigning a NEW primary broker
            run(session, "CALL custom.addNode('NODE_broker-node-2', 'Broker Server 2')");
            run(session, "CALL custom.addBroker('Broker Server 2', true)");

            Thread.sleep(1000);

            bCount = session.run("MATCH (b:Broker {id: 'BROKER_NODE_broker-node-1'}) RETURN count(b) AS c").next().get("c").asLong();
            assertTrue(bCount == 0, "[SCENARIO 6] Old Broker 1 was Garbage Collected successfully");

            primaryBrokers = session.run("MATCH (b:Broker {isPrimary: true}) RETURN count(b) AS c").next().get("c").asLong();
            assertTrue(primaryBrokers == 1, "[SCENARIO 6] Exactly 1 primary broker exists");

            long newRoutes = session.run("MATCH (b:Broker {id: 'BROKER_NODE_broker-node-2'})-[:ROUTES]->(:Topic) RETURN count(*) AS c").next().get("c").asLong();
            assertTrue(newRoutes == 3, "[SCENARIO 6] All Topics were handed over successfully via APOC to Broker 2");
            
            run(session, "CALL custom.removeNode('NODE_broker-node-2')"); 

            // --- SCENARIO 7: Topic Rejection Without Alive Primary Broker ---
            System.out.println("\n--- SCENARIO 7: Topic Rejection Without Alive Primary Broker ---");
            // At this point all brokers are gone (broker-node-2 was just removed).
            // addTopic must return success=false when no alive primary broker exists.
            long topicBefore = session.run("MATCH (t:Topic) RETURN count(t) AS c").next().get("c").asLong();

            boolean topicSuccess = session.run("CALL custom.addTopic('TOPIC_should-fail', 'Should Fail Topic') YIELD success RETURN success")
                    .next().get("success").asBoolean();
            assertTrue(!topicSuccess, "[SCENARIO 7] addTopic must return false when no alive Primary Broker exists");

            long topicAfter = session.run("MATCH (t:Topic {id: 'TOPIC_should-fail'}) RETURN count(t) AS c").next().get("c").asLong();
            assertTrue(topicAfter == 0, "[SCENARIO 7] Topic node must NOT be created in DB");

            // Also test: zombie broker should NOT qualify as alive
            run(session, "CALL custom.addNode('NODE_zombie-host', 'Zombie Host')");
            run(session, "CALL custom.addBroker('Zombie Host', true)");
            run(session, "CALL custom.addTopic('TOPIC_pre-zombie', 'Pre Zombie Topic')");
            // Kill the node -> broker becomes zombie
            run(session, "CALL custom.removeNode('NODE_zombie-host')");
            
            long zombieExists = session.run("MATCH (b:Broker {isPrimary: true}) WHERE NOT EXISTS { MATCH (b)-[:RUNS_ON]->(:Node) } RETURN count(b) AS c").next().get("c").asLong();
            assertTrue(zombieExists == 1, "[SCENARIO 7] Zombie broker exists (no RUNS_ON)");
            
            boolean topicWithZombie = session.run("CALL custom.addTopic('TOPIC_zombie-reject', 'Zombie Reject') YIELD success RETURN success")
                    .next().get("success").asBoolean();
            assertTrue(!topicWithZombie, "[SCENARIO 7] addTopic must reject even when zombie primary exists");

            // --- SCENARIO 8: Non-Primary Broker Creation ---
            System.out.println("\n--- SCENARIO 8: Non-Primary Broker Creation ---");
            session.run("MATCH (n) DETACH DELETE n"); // clean slate

            run(session, "CALL custom.addNode('NODE_np-broker', 'Non-Primary Host')");
            run(session, "CALL custom.addBroker('Non-Primary Host', false)");

            long npBroker = session.run("MATCH (b:Broker {id: 'BROKER_NODE_np-broker'}) RETURN count(b) AS c").next().get("c").asLong();
            assertTrue(npBroker == 1, "[SCENARIO 8] Non-primary broker created successfully");

            boolean npIsPrimary = session.run("MATCH (b:Broker {id: 'BROKER_NODE_np-broker'}) RETURN b.isPrimary AS p").next().get("p").asBoolean();
            assertTrue(!npIsPrimary, "[SCENARIO 8] Broker isPrimary must be false");

            long npRoutes = session.run("MATCH (b:Broker {id: 'BROKER_NODE_np-broker'})-[:RUNS_ON]->(:Node {id: 'NODE_np-broker'}) RETURN count(*) AS c").next().get("c").asLong();
            assertTrue(npRoutes == 1, "[SCENARIO 8] Non-primary broker has RUNS_ON to its host node");

            // --- SCENARIO 9: removeTopic ---
            System.out.println("\n--- SCENARIO 9: removeTopic ---");
            session.run("MATCH (n) DETACH DELETE n"); // clean slate

            run(session, "CALL custom.addNode('NODE_rt-host', 'RT Host')");
            run(session, "CALL custom.addBroker('RT Host', true)");
            run(session, "CALL custom.addTopic('TOPIC_removable', 'Removable Topic')");

            long topicExists = session.run("MATCH (t:Topic {id: 'TOPIC_removable'}) RETURN count(t) AS c").next().get("c").asLong();
            assertTrue(topicExists == 1, "[SCENARIO 9] Topic created");

            long routeExists = session.run("MATCH (:Broker)-[:ROUTES]->(:Topic {id: 'TOPIC_removable'}) RETURN count(*) AS c").next().get("c").asLong();
            assertTrue(routeExists == 1, "[SCENARIO 9] Topic has ROUTES from broker");

            run(session, "CALL custom.removeTopic('TOPIC_removable')");

            long topicGone = session.run("MATCH (t:Topic {id: 'TOPIC_removable'}) RETURN count(t) AS c").next().get("c").asLong();
            assertTrue(topicGone == 0, "[SCENARIO 9] Topic deleted successfully");

            long routeGone = session.run("MATCH ()-[:ROUTES]->(:Topic {id: 'TOPIC_removable'}) RETURN count(*) AS c").next().get("c").asLong();
            assertTrue(routeGone == 0, "[SCENARIO 9] ROUTES relationship cleaned up with topic deletion");

            // --- SCENARIO 10: removeRunsOn → Orphan App Deletion ---
            System.out.println("\n--- SCENARIO 10: removeRunsOn → Orphan App Deletion ---");
            session.run("MATCH (n) DETACH DELETE n"); // clean slate

            run(session, "CALL custom.addNode('NODE_solo', 'Solo Server')");
            run(session, "CALL custom.addApplication('APP_lonely', 'LonelyApp')");
            run(session, "CALL custom.addRunsOn('LonelyApp', 'NODE_solo')");

            long lonelyBefore = session.run("MATCH (a:Application {name: 'LonelyApp'}) RETURN count(a) AS c").next().get("c").asLong();
            assertTrue(lonelyBefore == 1, "[SCENARIO 10] LonelyApp exists before removeRunsOn");

            // removeRunsOn should orphan and auto-delete the app
            boolean roResult = session.run("CALL custom.removeRunsOn('LonelyApp', 'NODE_solo') YIELD success, appDeleted RETURN appDeleted")
                    .next().get("appDeleted").asBoolean();
            assertTrue(roResult, "[SCENARIO 10] removeRunsOn reports appDeleted=true");

            long lonelyAfter = session.run("MATCH (a:Application {name: 'LonelyApp'}) RETURN count(a) AS c").next().get("c").asLong();
            assertTrue(lonelyAfter == 0, "[SCENARIO 10] LonelyApp auto-deleted after removeRunsOn (orphan)");

            // --- SCENARIO 11: removePublisher → DEPENDS_ON Cleanup ---
            System.out.println("\n--- SCENARIO 11: removePublisher → DEPENDS_ON Cleanup ---");
            session.run("MATCH (n) DETACH DELETE n"); // clean slate

            run(session, "CALL custom.addNode('NODE_pub-host', 'Pub Host')");
            run(session, "CALL custom.addBroker('Pub Host', true)");
            run(session, "CALL custom.addTopic('TOPIC_pub-test', 'Pub Test Topic')");

            run(session, "CALL custom.addApplication('APP_pub', 'PubApp')");
            run(session, "CALL custom.addRunsOn('PubApp', 'NODE_pub-host')");

            run(session, "CALL custom.addApplication('APP_sub', 'SubApp')");
            run(session, "CALL custom.addRunsOn('SubApp', 'NODE_pub-host')");

            run(session, "CALL custom.addPublishesTo('PubApp', 'TOPIC_pub-test')");
            run(session, "CALL custom.addSubscribesTo('SubApp', 'TOPIC_pub-test')");

            Thread.sleep(500);

            long depBefore = session.run("MATCH (:Application {name: 'SubApp'})-[:DEPENDS_ON {dependency_type: 'app_to_app'}]->(:Application {name: 'PubApp'}) RETURN count(*) AS c").next().get("c").asLong();
            assertTrue(depBefore == 1, "[SCENARIO 11] DEPENDS_ON exists before removePublisher");

            // Remove publisher side — dependency should be cleaned
            run(session, "CALL custom.removePublishesTo('PubApp', 'TOPIC_pub-test')");
            Thread.sleep(500);

            long depAfter = session.run("MATCH (:Application {name: 'SubApp'})-[:DEPENDS_ON]->(:Application {name: 'PubApp'}) RETURN count(*) AS c").next().get("c").asLong();
            assertTrue(depAfter == 0, "[SCENARIO 11] DEPENDS_ON deleted when publisher removed");

            // --- SCENARIO 12: Orphan Topic Cleanup (Safety Net) ---
            System.out.println("\n--- SCENARIO 12: Orphan Topic Cleanup (No Broker Left) ---");
            session.run("MATCH (n) DETACH DELETE n"); // clean slate

            // Create broker + topics
            run(session, "CALL custom.addNode('NODE_last-broker', 'Last Broker')");
            run(session, "CALL custom.addBroker('Last Broker', true)");
            run(session, "CALL custom.addTopic('TOPIC_orphan-1', 'Orphan Test 1')");
            run(session, "CALL custom.addTopic('TOPIC_orphan-2', 'Orphan Test 2')");

            long topicsCreated = session.run("MATCH (t:Topic) RETURN count(t) AS c").next().get("c").asLong();
            assertTrue(topicsCreated == 2, "[SCENARIO 12] Two topics created");

            // Remove the ONLY broker's host node. Broker becomes zombie (keeps ROUTES).
            // Then create a second node with NO broker — just remove the zombie broker manually
            // to simulate: no broker exists at all → Q_CLEANUP_ORPHAN_TOPICS should fire.
            // Actually, removeNode does: Q_CHECK_AND_DELETE_USELESS_BROKERS then Q_CLEANUP_ORPHAN_TOPICS.
            // The zombie keeps ROUTES so it won't be GC'd by useless check.
            // To trigger full orphan cleanup, we need ALL brokers gone.
            // Let's manually DETACH DELETE the zombie broker to simulate no broker scenario.
            run(session, "CALL custom.removeNode('NODE_last-broker')");

            // Zombie broker still holds routes
            long zombieBrokers = session.run("MATCH (b:Broker) RETURN count(b) AS c").next().get("c").asLong();
            assertTrue(zombieBrokers == 1, "[SCENARIO 12] Zombie broker survives removeNode");

            // Manually kill zombie to simulate total broker loss
            session.run("MATCH (b:Broker) DETACH DELETE b");

            // Topics are now orphans — but cleanup only runs inside procedures.
            // Create a dummy node and remove it to trigger removeNode's cleanup pipeline.
            run(session, "CALL custom.addNode('NODE_trigger-gc', 'GC Trigger')");
            run(session, "CALL custom.removeNode('NODE_trigger-gc')");

            long orphanTopics = session.run("MATCH (t:Topic) RETURN count(t) AS c").next().get("c").asLong();
            assertTrue(orphanTopics == 0, "[SCENARIO 12] Orphan topics cleaned up when no broker exists (safety net)");

            // --- SCENARIO 13: Orphan Topics Routed to New Primary ---
            System.out.println("\n--- SCENARIO 13: Orphan Topics → New Primary Broker Routing ---");
            session.run("MATCH (n) DETACH DELETE n"); // clean slate

            // Create broker, topics, then kill broker to make topics orphans
            run(session, "CALL custom.addNode('NODE_old-primary', 'Old Primary')");
            run(session, "CALL custom.addBroker('Old Primary', true)");
            run(session, "CALL custom.addTopic('TOPIC_survive-1', 'Survive 1')");
            run(session, "CALL custom.addTopic('TOPIC_survive-2', 'Survive 2')");

            // Kill node → broker becomes zombie with routes
            run(session, "CALL custom.removeNode('NODE_old-primary')");

            // Manually detach the ROUTES from the zombie so topics become true orphans
            session.run("MATCH (b:Broker)-[r:ROUTES]->(t:Topic) DELETE r");
            // Delete the zombie broker
            session.run("MATCH (b:Broker) DETACH DELETE b");

            // Topics now exist but have no broker routes (orphans)
            long orphansBefore = session.run("MATCH (t:Topic) WHERE NOT EXISTS { MATCH (:Broker)-[:ROUTES]->(t) } RETURN count(t) AS c").next().get("c").asLong();
            assertTrue(orphansBefore == 2, "[SCENARIO 13] Two orphan topics exist");

            // Register a new primary broker → Q_ROUTE_ALL_ORPHAN_TOPICS_TO_BROKER should bind them
            run(session, "CALL custom.addNode('NODE_new-primary', 'New Primary')");
            run(session, "CALL custom.addBroker('New Primary', true)");

            long routedTopics = session.run("MATCH (:Broker {id: 'BROKER_NODE_new-primary'})-[:ROUTES]->(t:Topic) RETURN count(t) AS c").next().get("c").asLong();
            assertTrue(routedTopics == 2, "[SCENARIO 13] All orphan topics routed to new Primary Broker");

            long stillOrphans = session.run("MATCH (t:Topic) WHERE NOT EXISTS { MATCH (:Broker)-[:ROUTES]->(t) } RETURN count(t) AS c").next().get("c").asLong();
            assertTrue(stillOrphans == 0, "[SCENARIO 13] No orphan topics remain");

            // Cleanup
            session.run("MATCH (n) DETACH DELETE n");
            
            System.out.println("\nAll Comprehensive Edge Cases Passed Successfully! The plugin correctly orchestrates graph structures and cleanup rules.");
        } catch (Exception e) {
            e.printStackTrace();
            org.junit.jupiter.api.Assertions.fail("Test execution failed: " + e.getMessage());
        }
    }
    
    private void run(Session session, String query) {
        session.run(query).consume();
    }
}
