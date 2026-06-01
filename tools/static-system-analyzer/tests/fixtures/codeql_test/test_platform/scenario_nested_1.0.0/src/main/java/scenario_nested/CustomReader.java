package scenario_nested;
public class CustomReader {
    public void custom_read(Object topic) {
        System.out.println("read: " + topic.getClass().getName());
    }
}
