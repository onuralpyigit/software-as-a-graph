package scenario_reflection;

import java.lang.reflect.Method;

/**
 * Scenario: Reflection-based call via Method.invoke().
 * main calls custom_write through reflection — CodeQL typically
 * does NOT resolve reflective calls, so this should NOT be detected.
 * Expected: (nothing detected)
 */
public class Main {
    public static void main(String[] args) throws Exception {
        CustomWriter writer = new CustomWriter();
        Method m = CustomWriter.class.getMethod("custom_write", Object.class);
        m.invoke(writer, new Alpha_class());
    }
}
