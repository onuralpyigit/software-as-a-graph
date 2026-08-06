# System/Subsystem Specification (SSS): System as a Graph (SaaG)

**Definition:** The System as a Graph (SaaG) Digital System Model is a static digital system model developed using an architectural digital twin approach, which models the structural and relational architecture of the system using a node-relationship representation, without actually running the system applications. In this model, system entities such as software units, middleware and communication services, processor/console units, topics, and messages are represented as nodes; the dependency, publishing, and consuming relationships between them are represented as relationships. The behavioral analysis dimension of the model is achieved not by running the components, but by overlaying Analytical Evaluation Data — derived from field records or the scenario generator — onto this model.

**Table 1. SSS Requirement Distribution**

| No | Component | Abbreviation | Number of Requirements |
|---|---|---|---|
| 1 | Model Setup Data Generation | SaaG-MSD | 19 |
| 2 | Scenario Generator | SaaG-SCG | 7 |
| 3 | Field Records Database | SaaG-FRD | 6 |
| 4 | Analytical Data Preparation | SaaG-ADP | 6 |
| 5 | Node-Relationship Based Core System Model | SaaG-CSM | 20 |
| 6 | Design Verification, Analysis and Evaluation | SaaG-VAE | 54 |
| **TOTAL** | | | **112** |

**Purpose:** The primary purpose of the model is architectural verification. Within this scope, structural/circular dependencies, publisher/consumer matches, the conformance of topic quality-of-service (QoS) parameters, the capacity conformance of hardware present in the system (CPU core count, RAM size, network bandwidth, etc.), and design patterns that violate architectural rules are statically audited at the design stage. Architectural verification also covers the detection of deviations (architectural drift) between the architecture envisioned in the design and the runtime structure observed in field data. In addition to architectural verification, the model allows for hypothetical scenario analyses; without breaking structural integrity, the user can create experimental design constructs by adding/removing nodes/relationships or changing attributes. In these hypothetical scenarios, the propagation of situations such as an entity becoming inactive, an increase in message density, or a narrowing of bandwidth to dependent entities, and their effects on the architecture, are evaluated analytically. Thus, the Digital System Model provides a repeatable verification environment aimed at predicting the architectural consequences of design decisions and changes before software units are even installed in the target environment.

---

## 1. Model Setup Data Generation

1. SaaG shall have a Model Setup Data Generation (SaaG-MSD) component that ensures the data underlying the creation of the Digital System Model is produced in a controlled, traceable, verifiable manner and can be transferred to model construction processes.

2. SaaG-MSD shall be able to access at least the following external data sources for the purpose of Model Setup Data generation, and shall manage the data obtained from these sources in a controlled, traceable manner:
   1. System configuration management database,
   2. System software units and installation scripts source code repository,
   3. System Software Units Package Repository,
   4. System Network Topology Data Source,

3. SaaG-MSD shall be able to obtain the system network topology data by one of the following methods:
   1. Automatically from an external data source (file, database, etc.) whose details will be determined during the critical design phase,
   2. Through the user manually entering the network topology parameters.

4. SaaG-MSD shall manage, for each data source, the source type, source name, access method, connection address, and the user information required for connection, as user-definable and savable configuration information.

5. SaaG-MSD shall carry out data acquisition operations in association with project information, platform information, and system version number.

6. SaaG-MSD shall be able to obtain current project information from the configuration management database.

7. SaaG-MSD shall be able to obtain platform information belonging to the selected project from the configuration management database.

8. SaaG-MSD shall be able to obtain system version information belonging to the selected project and platform from the configuration management database.

9. SaaG-MSD shall mark the currently effective version information within the system version information obtained from the configuration management database.

10. SaaG-MSD shall record, as the "Software Unit Version Inventory," the name and version information of the software units that will run in the system environment, according to the selected project, platform, and version information.

11. SaaG-MSD shall update and record the Software Unit Version Inventory using the candidate version of the software unit being evaluated for installation into the target environment together with the other software unit versions defined in the selected system version.

12. SaaG-MSD shall mark the data acquisition process with an error status upon detecting deficiency, access error, or format incompatibility in the data obtained from the configuration management database.

13. SaaG-MSD shall access and transfer into the system the source code, installation scripts, and configuration files of the software units within the scope of the Software Unit Version Inventory, via the source code repository.

14. SaaG-MSD shall record the file name, file path, package/version information, and update timestamp for each file obtained from the source code repository.

15. SaaG-MSD shall report the data acquisition process with a "missing data" status if any of the files that are mandatory to obtain from the source code repository — whose details will be determined during the critical design phase — are missing.

16. SaaG-MSD shall display and record the relevant error in the event of an access, authorization, or integrity error occurring in files obtained from the source code repository.

17. SaaG-MSD shall perform a mandatory-field-presence check, within the scope of model construction, for all source data received or manually entered.

18. SaaG-MSD shall record, for each piece of data that fails the check, the error reason, source name, source type, associated project/platform information, and error time.

19. SaaG-MSD shall prepare the source data that passes the verification checks for transfer to the model construction process, and shall save it as a Model Setup Data file.

---

## 2. Scenario Generator

1. SaaG shall have a Scenario Generator (SaaG-SCG) component capable of producing synthetic data based on scenario inputs determined by the user, without requiring field records.

2. SaaG-SCG shall serve as the data source for all system-wide simulation processes — whose details will be determined during the critical design phase — and shall produce the synthetic data to be used in simulation processes.

3. SaaG-SCG shall enable the user to determine the scenario scope, scenario type, time interval, data density, and the data types to be produced, as required for scenario generation.

4. SaaG-SCG shall be able to produce synthetic data in an equivalent structure conforming to the topic/message data schema, field naming, and value range constraints used by the software units, based on user inputs.

5. SaaG-SCG shall record the produced synthetic data together with the scenario name, production time, associated project information, platform information, and system version number.

6. SaaG-SCG shall record, in a traceable manner, the user inputs used in the production of the synthetic data.

7. SaaG-SCG shall prepare the produced synthetic data for transfer to the Analytical Data Preparation component.

---

## 3. Field Records Database

1. SaaG shall have a Field Records Database (SaaG-FRD) for the purpose of storing and managing, in a centralized manner, the system data records and telemetry data obtained via the system data recording mechanism from the platforms on which the system is installed, as "System Field Records."

2. SaaG-FRD shall enable the user to upload the telemetry and system data records obtained from the system field environment into the Field Records Database in a controlled, traceable manner; it shall record the uploaded records in association with the relevant project information, platform information, and system version number.

3. SaaG-FRD shall record the uploaded System Field Records, in a traceable manner, together with the record source, upload time, and the associated project, platform, and system version information.

4. SaaG-FRD shall enable the user to list, search, and select the existing System Field Records according to criteria such as project, platform, system version, record source, or upload time.

5. SaaG-FRD shall report and record any format incompatibility, integrity error, or missing field conditions detected during upload.

6. SaaG-FRD shall operate on storage hardware with a disk capacity whose details will be determined during the critical design phase.

---

## 4. Analytical Data Preparation

1. SaaG shall have an Analytical Data Preparation (SaaG-ADP) component that ensures the Analytical Evaluation Data — to be used in analysis, verification, and simulation processes — is prepared in a controlled, traceable, verifiable manner and can be transferred to the Core System Model.

2. SaaG-ADP shall be able to obtain the System Field Records to be used in preparing the Analytical Evaluation Data from the Field Records Database.

3. SaaG-ADP shall be able to obtain the synthetic data produced by the Scenario Generator as data required to create the Analytical Evaluation Data.

4. SaaG-ADP shall process the System Field Records or the synthetic data supplied by the Scenario Generator, associate them appropriately, and produce the Analytical Evaluation Data — whose details will be determined during the critical design phase — which shall then be transmitted to the Core System Model.

5. SaaG-ADP shall report and record the detection of format incompatibility or unreadable data in the System Field Records.

6. SaaG-ADP shall report and record the detection of format incompatibility, unreadable data, or missing fields in the synthetic data supplied by the Scenario Generator.

---

## 5. Node-Relationship Based Core System Model

1. SaaG shall have a Node-Relationship Based Core System Model (SaaG-CSM) component that uses the Model Setup Data to construct the structural and relational representation of the system in a node-relationship structure; and that matches the Analytical Evaluation Data with the relevant system entities and the connections between them in the model, making it usable in static analysis, verification, and simulation processes.

2. SaaG-CSM shall be able to accept, as input, the Model Setup Data produced by the Model Setup Data Generation component.

3. SaaG-CSM shall perform format, schema, integrity, and mandatory field checks on the Model Setup Data before the construction of the Core System Model.

4. SaaG-CSM shall convert the Model Setup Data that passes the checks into a node-relationship based Core System Model.

5. SaaG-CSM shall create the Core System Model in association with the relevant project, platform, and system version information.

6. SaaG-CSM shall represent, as nodes in the node-relationship structure, at least the following structural system entities found in the Model Setup Data:
   1. System,
   2. Software Segment,
   3. Computer Software Configuration Item (CSCI),
   4. Computer Software Component (CSC),
   5. Computer Software Unit (CSU),
   6. Role,
   7. Topic,
   8. Message,
   9. Operator Console and Processor Units,
   10. Network components,
   11. Middleware Services,
   12. Services belonging to Communication Technologies.

7. SaaG-CSM shall represent, as relationships in the node-relationship structure, at least the following relationship types between structural system entities:
   1. Running on Operator Console and Processor Units,
   2. Using Middleware and Communication Services,
   3. Publishing data,
   4. Consuming data,
   5. Being dependent on a library or software unit,
   6. Assignment of a software unit to a role.

8. SaaG-CSM shall represent the processor core allocation (CPU allocation), operating system settings, and runtime environment configurations (JVM, etc.) belonging to the system's software units as queryable attributes on the node-relationship structure.

9. SaaG-CSM shall report and record missing entity and invalid relationship errors detected during the construction of the Core System Model.

10. SaaG-CSM shall be able to accept, as input, the Analytical Evaluation Data produced by the Analytical Data Preparation component.

11. SaaG-CSM shall associate the Analytical Evaluation Data with the relevant project, platform, system version, and Core System Model; it shall match the record, telemetry, and synthetic data found in the data with the relevant nodes and relationships, and bind it to the node-relationship structure.

12. SaaG-CSM shall preserve information on whether the Analytical Evaluation Data was produced using System Field Records or synthetic data supplied by the Scenario Generator.

13. SaaG-CSM shall bind the Analytical Evaluation Data to the model without altering the nodes and relationships in the Core System Model, and shall ensure that the Core System Model data and the Analytical Evaluation Data are managed in a manner that keeps them separable from one another.

14. SaaG-CSM shall report and record any node or relationship records for which no counterpart can be found in the Analytical Evaluation Data.

15. SaaG-CSM shall record the Model Setup Data file used for the created Core System Model, the model creation time, project information, platform information, system version number, and model status.

16. SaaG-CSM shall make the Core System Model available for use by the Design Verification, Analysis and Evaluation component.

17. SaaG-CSM shall enable the Design Verification, Analysis and Evaluation component to access the nodes, relationships, and the Analytical Evaluation Data associated with them.

18. SaaG-CSM shall handle read/write operations performed concurrently by multiple user sessions on the same Core System Model without compromising model integrity or the consistency of query results.

19. SaaG-CSM shall execute the operations in the production deployment pipeline and the analysis and simulation operations of a number of users — to be determined during the critical design phase — concurrently and independently of one another, preventing the operations from affecting each other.

20. SaaG-CSM shall create a new, process-specific Core System Model using the candidate version of the software unit being evaluated for installation into the target environment together with the other software units in the target system version.

---

## 6. Design Verification, Analysis and Evaluation

1. SaaG shall have a Design Verification, Analysis and Evaluation (SaaG-VAE) component that enables the user to interact directly with the system components and to perform design verification, static analysis, and evaluation operations on the model.

2. SaaG-VAE shall be able to interact with the Model Setup Data Generation, Scenario Generator, Analytical Data Preparation, and Core System Model components.

3. SaaG-VAE shall authenticate the username and password information of users wishing to access the system via a defined LDAP directory service, and shall allow only users who successfully authenticate to access the system within the scope of their authorizations.

4. SaaG-VAE shall enable the user to select the project, platform, and system version on which operations will be performed, and shall distinctly display the currently effective system version for the project and platform.

5. SaaG-VAE shall list, for the user, the Model Setup Data files belonging to the selected project, platform, and system version, and shall enable the user to select the file to be used.

6. SaaG-VAE shall enable the user to start the Model Setup Data production process and to monitor the status of the process as one of in progress, successful, or failed.

7. SaaG-VAE shall continuously and traceably display to the user the accessibility status of all data sources used.

8. SaaG-VAE shall display to the user the missing data, access, authorization, format, or integrity errors detected during Model Setup Data production.

9. SaaG-VAE shall enable the user to start the process of creating the Core System Model using the selected Model Setup Data, and to monitor the result of the operation as one of successful or failed.

10. SaaG-VAE shall enable the user to determine the data source to be used for creating the Analytical Evaluation Data as one of the following options:
    1. System Field Records,
    2. Synthetic data supplied by the Scenario Generator.

11. SaaG-VAE shall enable the user to select the records to be used, in the case where System Field Records are to be used.

12. SaaG-VAE shall enable the user to determine the inputs relating to scenario scope, scenario type, time interval, data density, and the data types to be produced, in the case where synthetic data are to be used.

13. SaaG-VAE shall enable the user to start and track the synthetic data production process, and to view errors occurring during production.

14. SaaG-VAE shall enable the user to start and track the Analytical Evaluation Data production process, and to view errors occurring during production.

15. SaaG-VAE shall perform design verification and analysis operations without altering the nodes and relationships in the Core System Model.

16. SaaG-VAE shall display to the user the project, platform, and system version information associated with the Analytical Evaluation Data bound to the Core System Model, and shall report the matching status of the record, telemetry, and synthetic data found in the data with the nodes and relationships.

17. SaaG-VAE shall enable the user to perform structural changes — such as adding/removing nodes, adding/removing relationships, and updating node/relationship attributes — on a working model derived from the Core System Model without breaking its structural integrity; and shall enable design verification and analysis operations to be carried out on the updated working model.

18. SaaG-VAE shall be able to perform analyses solely on the Core System Model without using Analytical Evaluation Data.

19. SaaG-VAE shall be able to perform, on the Core System Model, the analysis of structural dependencies, communication connections, and runtime environment relationships between system entities.

20. SaaG-VAE shall verify, on the Core System Model, the conformance of topic data transmission quality-of-service parameters to rules to be determined during the critical design phase, and shall detect incompatibilities in relation to at least the following parameters:
    1. Durability,
    2. Reliability,
    3. Lifespan,
    4. Transport Priority.

21. SaaG-VAE shall verify topic data publisher and data consumer matches on the Core System Model, and shall detect at least the following incompatibilities:
    1. A topic with no data publisher,
    2. A topic with no data consumer,
    3. Topics defined with the same name having content definitions that differ from one another.

22. SaaG-VAE shall verify, on the Core System Model, the mutual consistency of source, destination, message, and communication direction information in external-to-middleware communications carried out via communication services to be determined during the critical design phase.

23. SaaG-VAE shall analyze, on the Core System Model, the conformance of the distribution of the system's software units across Operator Console and Processor Units to load balancing rules to be determined during the critical design phase.

24. SaaG-VAE shall verify, on the Core System Model, the conformance of the processor core allocation made to the system's software units to rules to be determined during the critical design phase, and shall detect at least the following incompatibilities:
    1. The total number of cores allocated on a Processor Unit exceeding the available core capacity,
    2. The same cores being allocated to multiple applications in a conflicting manner,
    3. Applications required to run with high performance not having dedicated cores allocated to them.

25. SaaG-VAE shall audit, on the Core System Model, the conformance of the operating system settings running on processor/console units to rules to be determined during the critical design phase and to the processor core allocation made.

26. SaaG-VAE shall verify, on the Core System Model, the conformance of the memory allocation parameters in the runtime environment configurations of the system's software units to rules to be determined during the critical design phase.

27. SaaG-VAE shall detect, on the Core System Model, situations that could cause resource contention and bottlenecks arising from inconsistencies among processor core allocation, operating system settings, and runtime environment configurations.

28. SaaG-VAE shall detect circular dependencies between the system's software units on the Core System Model.

29. SaaG-VAE shall detect, on the Core System Model, disconnected, missing, invalid, or unmatched structural relationships between the nodes within the Core System Model.

30. SaaG-VAE shall detect, on the Core System Model, design patterns that violate architectural rules to be determined during the critical design phase.

31. SaaG-VAE shall be able to perform analyses using Analytical Evaluation Data produced from synthetic data supplied by the Scenario Generator.

32. SaaG-VAE shall analyze the message flow direction, message count, data volume, and messaging frequency between nodes using Analytical Evaluation Data produced from synthetic data supplied by the Scenario Generator.

33. SaaG-VAE shall be able to evaluate the effects on the Core System Model of a node or relationship becoming inactive, using Analytical Evaluation Data produced from synthetic data supplied by the Scenario Generator.

34. SaaG-VAE shall be able to perform design-time traffic analysis using Analytical Evaluation Data produced from synthetic data supplied by the Scenario Generator, and shall be able to evaluate, within the scope of the effects of load conditions created within the simulation on system entities and relationships, at least the following situations:
    1. An increase in Topic/Message density,
    2. A change in Topic/Message publishing or consumption behavior.

35. SaaG-VAE shall, using Analytical Evaluation Data produced from synthetic data supplied by the Scenario Generator, determine the propagation of fault, load, communication interruption, or bandwidth-narrowing conditions created within the simulation onto dependent nodes; and shall detect the directly or indirectly affected nodes/relationships and the propagation path followed by the effect.

36. SaaG-VAE shall, using Analytical Evaluation Data produced from synthetic data supplied by the Scenario Generator, determine the system entities with the highest resource usage or the most intensive messaging as a result of the simulation to be performed, and shall present these to the user as summary evaluation indicators.

37. SaaG-VAE shall be able to perform analyses using Analytical Evaluation Data produced from System Field Records.

38. SaaG-VAE shall be able to perform analyses on the Core System Model, using Analytical Evaluation Data produced from System Field Records, specifically on at least the following topics:
    1. Operational and health status,
    2. Processor, memory, storage, and network usage values,
    3. Error, warning, restart, and timeout information,
    4. Message flow direction, message count, data volume, and messaging frequency,
    5. Communication latency, message loss, and successful transmission rates,
    6. Topic publishing and consumption activities.

39. SaaG-VAE shall compare the nodes and relationships in the Model Setup Data with the runtime system entities and relationships observed in the Analytical Evaluation Data produced from System Field Records, and shall detect at least the following situations:
    1. System entities and relationships present in the Model Setup Data but not observed in the runtime data,
    2. System entities and relationships not present in the Model Setup Data but observed in the runtime data,
    3. System entities and relationships showing incompatibility between the Model Setup Data and the runtime data.

40. SaaG-VAE shall analyze the event records associated with the nodes and relationships found in the Analytical Evaluation Data produced from System Field Records.

41. SaaG-VAE shall, using Analytical Evaluation Data produced from System Field Records, determine the system entities with the highest resource usage or the most intensive messaging as a result of the analysis, and shall present these to the user as summary evaluation indicators.

42. SaaG-VAE shall classify design verification and analysis results as one of "conforming" or "non-conforming," according to rules/metrics to be determined during the critical design phase.

43. SaaG-VAE shall enable the user to search for a system entity or relationship on the node-relationship structure; to filter the results by type, project, platform, system version, or software unit information; and to perform visual zoom in, zoom out, pan, and node/relationship selection and attribute display operations.

44. SaaG-VAE shall present to the user each finding detected in the analysis results together with at least the following information:
    1. Finding identifier,
    2. Finding type,
    3. Finding description,
    4. Affected system entity or relationship,
    5. Related verification rule or acceptance criterion,
    6. Data or evidence supporting the finding,
    7. The severity level of the finding, expressed as one of informational, low, medium, high, or critical.

45. SaaG-VAE shall record and display to the user the cause-and-effect relationship between related findings detected within the scope of the same operation.

46. SaaG-VAE shall enable the user to sort and filter findings by operation type, evaluation result, finding type, severity level, project, platform, system version, or affected nodes.

47. SaaG-VAE shall record the error cause, the stage at which the operation was interrupted, and the error time occurring during a design verification, analysis, or simulation operation.

48. SaaG-VAE shall be able to record the scenario name, scenario inputs, data production time, and the associated project, platform, and system version information used in simulation operations.

49. SaaG-VAE shall generate a summary or detailed system report of design verification, analysis, and simulation results in an exportable file format whose details will be determined during the critical design phase, and shall ensure that the reports contain at least the following information:
    1. Project information,
    2. Platform information,
    3. System version information,
    4. The Core System Model used,
    5. The Analytical Evaluation Data used and its data source,
    6. Operation identifier and operation type,
    7. Operation start and end time,
    8. Evaluation result,
    9. Findings detected,
    10. Affected nodes and relationships,
    11. Severity levels,
    12. Additional information relating to the findings.

50. SaaG-VAE shall also accept analysis requests — made via user interfaces — through Build Automation Tools and a Command Line Interface (CLI); shall present status information on ongoing operations to users accessing the system and to automation clients (e.g., Jenkins); and shall ensure that analysis operations are carried out concurrently and independently of one another.

51. SaaG-VAE shall analyze the suitability of a software unit for installation into the target environment under at least the following evaluation headings:
    1. Structural and architectural conformance,
    2. Interface, topic, and communication conformance,
    3. Dependency and integration conformance,
    4. Resource and performance sufficiency.

52. SaaG-VAE shall define each control rule used in the installation suitability evaluation with a rule identifier, evaluation heading, severity level, weight value, acceptance criterion, and blocking status; and shall classify and score the conformance categories and scoring method belonging to the rule results in a manner whose details will be determined during the critical design phase.

53. SaaG-VAE shall, upon detecting a finding with a critical severity level or a violation of a control rule defined as blocking in the evaluation profile, determine the installation result for the target environment as "non-conforming" independently of the overall conformance score, and shall transmit the decision information preventing the continuation of the production deployment pipeline to the automation client.

54. SaaG-VAE shall execute installation suitability evaluations initiated for one or more software units within the scope of the production deployment pipeline using independent operation identifiers from one another; and shall present, for each software unit, a separate conformance score, score class, blocking findings, and installation decision, as well as the aggregate operation result, in a machine-processable format to the automation client.
