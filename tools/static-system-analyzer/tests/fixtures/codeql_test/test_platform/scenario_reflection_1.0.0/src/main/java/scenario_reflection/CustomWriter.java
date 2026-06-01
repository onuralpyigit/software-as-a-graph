package scenario_reflection;
public class CustomWriter {
    public void custom_write(Object topic) {
        System.out.println("write: " + topic.getClass().getName());
    }
}
