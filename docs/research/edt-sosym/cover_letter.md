**Submission for Theme Section: Engineering of Digital Twins**

Dear Editors of the EDTConf Theme Section,

We submit the manuscript *"Engineering an Architectural Digital Twin for Distributed Publish–Subscribe Systems: One Graph Model for Analysis, Simulation, and an Executable Container Testbed"* for consideration as a Theme Section Paper in *Software and Systems Modeling*.

**Engineering challenge.** Publish–subscribe systems (DDS, ROS 2, Kafka, MQTT) fail through dependencies that exist only in the assembled system. Engineers need an executable counterpart of the architecture before deployment, and a way to check it against the running system afterwards. Such architectural twins are still built ad hoc, and their fidelity is seldom measured.

**Engineering method.** The manuscript presents a model-driven method in which one typed graph metamodel is the source of every twin artefact: analysis views, four failure simulators, and a generated Docker and Cyclone DDS counterpart that plays the physical role. The method also contributes:
- a model-to-system transformation, with a single fault vocabulary for simulation and chaos;
- a system-to-model calibration path, with an identifiability analysis;
- a fidelity-assessment method that separates genuine evidence from agreement that holds by construction, supported by a pre-registered study protocol.

We position the system on the digital model, shadow and twin maturity ladder, and claim only the levels that are built: a digital model with a generated counterpart and a batch digital shadow. Every fidelity claim is bounded to middleware, timing and transport behaviour under declared application behaviour.

**Fit to the call.** Each contribution addresses topics of the theme section:
- the single metamodel with executable consistency checks: modeling methods for digital twins, and model consistency;
- the model-to-system transformation: DT modeling and simulation, and, with the change gate and the cost ladder, DevOps for digital twins;
- the calibration path and identifiability analysis: combining models and data, and uncertainty;
- the fidelity-assessment method: quality assurance and evaluation of digital twins, and fidelity.

**Related work by the authors.** A reduced form of the dependency derivation appeared in IEEE RASSE 2025. A companion manuscript on the simulators as ranking oracles for criticality prediction is under review at the *Journal of Systems and Software*. This manuscript cites both and reproduces no results from either. A tool paper on the framework's modeling language, quality diagnosis and refactoring is in preparation for *Software and Systems Modeling*; it shares the metamodel description but none of the twin method, and this manuscript reuses no text or results from it. The testbed, calibration, fidelity method and maturity analysis are reported here for the first time. The manuscript is not under consideration elsewhere.

We would be glad to present the work at EDTConf if it is accepted.

Sincerely,
Ibrahim Onuralp Yigit and Feza Buzluca
[Affiliation, address, e-mail]
