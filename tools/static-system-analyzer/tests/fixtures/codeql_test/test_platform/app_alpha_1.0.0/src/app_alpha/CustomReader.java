package app_alpha;

public class CustomReader {
    public void custom_read(Object topic) {
        System.out.println("Reading topic: " + topic.getClass().getName());
    }
}
