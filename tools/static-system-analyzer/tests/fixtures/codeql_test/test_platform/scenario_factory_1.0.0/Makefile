-include include/Makefile_java.mk

.PHONY: all clean regenerate_code

regenerate_code: clean all

clean:
	mvn clean -q

all:
	mvn compile -q
