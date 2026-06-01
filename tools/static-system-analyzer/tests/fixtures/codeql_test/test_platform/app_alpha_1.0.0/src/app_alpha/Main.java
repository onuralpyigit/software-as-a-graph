package app_alpha;

public class Main {
    public static void main(String[] args) {
        // Direct write — topic from Alpha_class arg
        CustomWriter writer = new CustomWriter();
        writer.custom_write(new Alpha_class());

        // Read via helper_lib (call chain: main → HelperLib.process → CustomReader.custom_read)
        helper_lib.HelperLib helper = new helper_lib.HelperLib();
        helper.process();

        // Direct read
        CustomReader reader = new CustomReader();
        reader.custom_read(new Beta_class());
    }
}
