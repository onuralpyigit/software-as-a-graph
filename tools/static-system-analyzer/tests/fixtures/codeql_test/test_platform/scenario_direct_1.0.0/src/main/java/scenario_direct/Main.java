package scenario_direct;

/**
 * Scenario: Direct call from main to custom_write / custom_read.
 * Expected: Alpha=pub, Beta=sub
 */
public class Main {
    public static void main(String[] args) {
        CustomWriter writer = new CustomWriter();
        writer.custom_write(new Alpha_class());

        CustomReader reader = new CustomReader();
        reader.custom_read(new Beta_class());
    }
}
