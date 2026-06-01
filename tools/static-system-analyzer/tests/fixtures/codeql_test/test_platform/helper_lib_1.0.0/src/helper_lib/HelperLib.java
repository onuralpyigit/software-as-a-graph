package helper_lib;

import app_alpha.CustomReader;

public class HelperLib {
    public void process() {
        CustomReader reader = new CustomReader();
        reader.custom_read(new Gamma_class());
    }

    public static void main(String[] args) {
        // Lib's own main — direct read of Gamma
        HelperLib lib = new HelperLib();
        lib.process();
    }
}
