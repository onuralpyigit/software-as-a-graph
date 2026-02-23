# Auto-detect Maven project directories (each containing a pom.xml)
MODULES := $(sort $(dir $(wildcard */pom.xml)))

.PHONY: all clean jar test

# Default target
all: jar

# Clean all modules
clean:
	@for mod in $(MODULES); do \
		echo "Cleaning $$mod..."; \
		cd $$mod && mvn clean && cd ..; \
	done

# Build JAR for all modules
jar:
	@for mod in $(MODULES); do \
		echo "Building $$mod..."; \
		cd $$mod && mvn clean package -DskipTests && cd ..; \
	done

# Run tests for all modules
test:
	@for mod in $(MODULES); do \
		echo "Testing $$mod..."; \
		cd $$mod && mvn test && cd ..; \
	done

