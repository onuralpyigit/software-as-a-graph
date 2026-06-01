package scenario_polymorphism;

/**
 * Polymorphic implementation — delegates to CustomWriter.custom_write()
 * with a specific topic type created internally.
 */
public class ConcreteWriter implements WriterInterface {
    private final CustomWriter cw = new CustomWriter();

    @Override
    public void write() {
        cw.custom_write(new Alpha_class());
    }
}
