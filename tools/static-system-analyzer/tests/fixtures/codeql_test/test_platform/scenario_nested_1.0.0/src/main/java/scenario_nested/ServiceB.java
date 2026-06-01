package scenario_nested;

public class ServiceB {
    public void handle() {
        new ServiceC().execute();
    }
}
