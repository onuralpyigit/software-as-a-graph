"""
saag.adapters — importers for external architecture descriptions.

The one real member is :mod:`saag.adapters.realworld_adapter`, which transcribes
open-source repositories (ROS 2 launch graphs, Docker Compose / Kubernetes
manifests, EdgeX and Home Assistant configurations) into the topology JSON the
rest of the pipeline consumes.

This module used to also re-export ``Neo4jRepository``, ``create_repository`` and
``config`` under a docstring reading "Deprecated: use src.infrastructure" -- a
package that does not exist and never did under that name. The re-export was
nonetheless load-bearing: four API routers, ``api/models.py`` and
``saag/simulation/traffic_simulator.py`` imported persistence through it. They
now import from :mod:`saag.infrastructure` directly, which is where those live.
"""
