Name:           genieus
Version:        1.0.0
Release:        1%{?dist}
Summary:        Graph-Based Distributed System Analysis Platform
License:        MIT
URL:            https://github.com/oersen/software-as-a-graph

Source0:        genieus-1.0.0.tar.gz
Source1:        genieus.service

BuildArch:      x86_64
Requires:       /usr/bin/docker
Requires:       systemd

%description
Genieus predicts which components in a distributed publish-subscribe system
will cause the most damage when they fail, using topological analysis of
the system architecture modeled as a weighted directed graph.

Includes Neo4j, FastAPI backend, and Next.js frontend in a single container.

%prep

%build

%install
install -d %{buildroot}/opt/genieus
install -m 644 %{SOURCE0} %{buildroot}/opt/genieus/genieus-1.0.0.tar.gz
install -d %{buildroot}%{_unitdir}
install -m 644 %{SOURCE1} %{buildroot}%{_unitdir}/genieus.service

%pre
FAIL=0

# Check Docker
if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is not installed."
    echo "  Install with: sudo dnf install docker-ce  (or: sudo yum install docker)"
    FAIL=1
elif ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker daemon is not running."
    echo "  Start with:   sudo systemctl start docker"
    FAIL=1
fi

# Check systemd
if ! command -v systemctl >/dev/null 2>&1; then
    echo "ERROR: systemd is not available on this system."
    FAIL=1
fi

# Check ports
for PORT in 7000 7474 7687 8000; do
    if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
        echo "WARNING: Port ${PORT} is already in use. Genieus may fail to start."
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "Prerequisites not met. Aborting installation."
    exit 1
fi

echo "All prerequisites satisfied."

%post
echo "Loading Genieus Docker image..."
docker load -i /opt/genieus/genieus-1.0.0.tar.gz
systemctl daemon-reload
systemctl enable --now genieus.service
echo ""
echo "Genieus is starting up. Service endpoints will be available shortly:"
echo "  Frontend:       http://localhost:7000"
echo "  Backend API:    http://localhost:8000"
echo "  API Docs:       http://localhost:8000/docs"
echo "  Neo4j Browser:  http://localhost:7474"
echo ""
echo "Check status: systemctl status genieus"
echo "View logs:    docker logs -f genieus"

%preun
if [ "$1" = "0" ]; then
    systemctl disable --now genieus.service 2>/dev/null || true
    docker rm -f genieus 2>/dev/null || true
fi

%postun
if [ "$1" = "0" ]; then
    docker rmi genieus:1.0.0 2>/dev/null || true
    rm -rf /opt/genieus
    systemctl daemon-reload
fi

%files
/opt/genieus/genieus-1.0.0.tar.gz
%{_unitdir}/genieus.service

%changelog
* Mon Feb 24 2026 Genieus Maintainer <maintainer@genieus.dev> - 1.0.0-1
- Initial RPM package with bundled Docker image
