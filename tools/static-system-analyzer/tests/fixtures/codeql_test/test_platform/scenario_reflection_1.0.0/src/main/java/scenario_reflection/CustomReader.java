package scenario_reflection;
public class CustomReader {
    public void custom_read(Object topic) {
        System.out.println("read: " + topic.getClass().getName());
    }
}
