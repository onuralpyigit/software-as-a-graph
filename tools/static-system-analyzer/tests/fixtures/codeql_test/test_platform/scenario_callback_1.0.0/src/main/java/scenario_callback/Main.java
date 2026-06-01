package scenario_callback;

/**
 * Scenario: Callback via functional interface (Runnable lambda).
 * main creates a lambda that calls custom_write, then executes it.
 * Expected: Alpha=pub
 */
public class Main {
    public static void main(String[] args) {
        CustomWriter writer = new CustomWriter();
        Runnable callback = () -> writer.custom_write(new Alpha_class());
        callback.run();
    }
}
