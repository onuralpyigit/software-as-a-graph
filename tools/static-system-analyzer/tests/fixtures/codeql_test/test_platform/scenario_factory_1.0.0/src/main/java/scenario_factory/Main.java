package scenario_factory;

/**
 * Scenario: Factory pattern — factory creates writer, main invokes it.
 * main → factory.create() → .custom_write(new Alpha_class())
 * main → factory.create() → .custom_write(new Beta_class())
 * Expected: Alpha=pub, Beta=pub
 */
public class Main {
    public static void main(String[] args) {
        WriterFactory factory = new WriterFactory();
        factory.create().custom_write(new Alpha_class());
        factory.create().custom_write(new Beta_class());
    }
}
