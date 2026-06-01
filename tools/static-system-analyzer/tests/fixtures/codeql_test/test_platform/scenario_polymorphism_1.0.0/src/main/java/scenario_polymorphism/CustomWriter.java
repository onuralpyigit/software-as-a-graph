package scenario_polymorphism;
public class CustomWriter {
    public void custom_write(Object topic) {
        System.out.println("write: " + topic.getClass().getName());
    }
}
