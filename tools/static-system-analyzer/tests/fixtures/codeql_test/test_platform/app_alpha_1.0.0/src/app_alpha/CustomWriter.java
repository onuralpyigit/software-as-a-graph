package app_alpha;

public class CustomWriter {
    public void custom_write(Object topic) {
        System.out.println("Writing topic: " + topic.getClass().getName());
    }
}
