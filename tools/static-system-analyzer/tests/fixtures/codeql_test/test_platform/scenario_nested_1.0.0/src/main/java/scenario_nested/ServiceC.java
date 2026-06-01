package scenario_nested;

public class ServiceC {
    public void execute() {
        CustomWriter writer = new CustomWriter();
        writer.custom_write(new Alpha_class());
    }
}
