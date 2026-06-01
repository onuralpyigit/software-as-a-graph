package scenario_polymorphism;

/**
 * Scenario: Polymorphic dispatch through interface.
 * main → WriterInterface.write() → ConcreteWriter.write() → custom_write(Alpha_class)
 * Expected: Alpha=pub
 */
public class Main {
    public static void main(String[] args) {
        WriterInterface w = new ConcreteWriter();
        w.write();
    }
}
