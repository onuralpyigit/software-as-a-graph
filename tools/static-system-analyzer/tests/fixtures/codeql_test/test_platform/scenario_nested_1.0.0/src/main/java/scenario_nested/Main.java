package scenario_nested;

/**
 * Scenario: Deep nested call chain.
 * main → ServiceA.process() → ServiceB.handle() → ServiceC.execute() → custom_write()
 * Expected: Alpha=pub
 */
public class Main {
    public static void main(String[] args) {
        new ServiceA().process();
    }
}
