package scenario_nested;

public class ServiceA {
    public void process() {
        new ServiceB().handle();
    }
}
