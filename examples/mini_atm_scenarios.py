"""
Proof-of-concept incident scenarios on the mini-ATM topology.

One shared catalogue, read by run_simulator_walkthrough.py (the Failure Lab,
simulated) and run_testbed_walkthrough.py (the same incidents on containers),
so the two pages always show the same incidents.

``faults`` are `FaultSpec` dicts (saag/simulation/runtime/faults.py), which
drive both MessageFlowSimulator and the testbed. ``fs`` is how FailureSimulator
expresses the incident -- targets and a `FailureMode` value -- or None where it
cannot. ``fs_note`` says what that approximation leaves out.
"""

DURATION_S = 40.0

SCENARIOS = [
    {"id": "S1", "group": "Degraded behaviour", "title": "Tracker starved of CPU",
     "story": "From 20 s tracker takes four times as long to process each radar plot.",
     "faults": [{"target": "A2", "mode": "slowdown", "start_s": 20, "params": {"k": 4.0}}],
     "fs": {"targets": ["A2"], "mode": "degraded"}, "fs_note": "DEGRADED: tracker at half capacity"},
    {"id": "S2", "group": "Degraded behaviour", "title": "Radar dropping half its plots",
     "story": "From 20 s radar loses each plot it emits with probability 0.5.",
     "faults": [{"target": "A1", "mode": "omission", "start_s": 20, "params": {"p": 0.5}}],
     "fs": {"targets": ["A1"], "mode": "degraded"}, "fs_note": "DEGRADED: radar at half capacity"},
    {"id": "S3", "group": "Degraded behaviour", "title": "Radar babbling",
     "story": "From 20 s radar publishes at five times its declared rate and floods tracker.",
     "faults": [{"target": "A1", "mode": "babbling", "start_s": 20, "params": {"k": 5.0}}],
     "fs": None, "fs_note": "FailureSimulator has no overload mode (OVERLOAD is declared but not implemented)"},
    {"id": "S4", "group": "Network trouble", "title": "Flapping link between the hosts",
     "story": "From 15 s the sensor-host to ops-host link drops for about 0.5 s every 2 s.",
     "faults": [{"target": "N1|N2", "mode": "link_flap", "start_s": 15,
                 "params": {"mean_good_s": 2.0, "mean_bad_s": 0.5}}],
     # A cut link only gates traffic when the engines model the network.
     "network_gating": True,
     "fs":{"targets": ["N1"], "mode": "partition"}, "fs_note": "PARTITION of sensor-host, held for good"},
    {"id": "S5", "group": "Network trouble", "title": "Ops-broker hangs for ten seconds",
     "story": "The ops-broker relay stops serving at 20 s and resumes at 30 s; messages wait.",
     "faults": [{"target": "B2", "mode": "freeze", "start_s": 20, "duration_s": 10}],
     "fs": {"targets": ["B2"], "mode": "crash"}, "fs_note": "as a permanent crash: FailureSimulator has no time"},
    {"id": "S6", "group": "Correlated failures", "title": "Both broker relays down",
     "story": "Both relays crash at 20 s, so tracks loses its redundancy and every topic its route.",
     "faults": [{"target": "B1", "mode": "crash", "start_s": 20}, {"target": "B2", "mode": "crash", "start_s": 20}],
     "fs": {"targets": ["B1", "B2"], "mode": "crash"}, "fs_note": "both crashed together"},
    {"id": "S7", "group": "Correlated failures", "title": "Radar and flight-plan lost together",
     "story": "The two independent data sources crash at 20 s.",
     "faults": [{"target": "A1", "mode": "crash", "start_s": 20}, {"target": "A4", "mode": "crash", "start_s": 20}],
     "fs": {"targets": ["A1", "A4"], "mode": "crash"}, "fs_note": "both crashed together"},
    {"id": "S8", "group": "Recovery and restarts", "title": "Sensor host reboots",
     "story": "Sensor-host goes down at 15 s and is back at 25 s; its applications restart fresh.",
     "faults": [{"target": "N1", "mode": "transient_crash", "start_s": 15, "duration_s": 10}],
     "fs": {"targets": ["N1"], "mode": "crash"}, "fs_note": "as a permanent crash: FailureSimulator has no time"},
    {"id": "S9", "group": "Recovery and restarts", "title": "Rolling relay restart",
     "story": "Sensor-broker restarts from 10 to 18 s, then ops-broker from 22 to 30 s.",
     "faults": [{"target": "B1", "mode": "transient_crash", "start_s": 10, "duration_s": 8},
                {"target": "B2", "mode": "transient_crash", "start_s": 22, "duration_s": 8}],
     "fs": None, "fs_note": "a sequence of outages has no single-shot equivalent"},
]
