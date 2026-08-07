package com.surish.ai;

import com.surish.ai.dataset.DatasetLoader;
import java.io.IOException;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {
        DatasetLoader loader = new DatasetLoader();

        try {
            String dataset = loader.load();
            System.out.println("Characters : " + dataset.length());
            System.out.println();
            System.out.println(dataset.substring(0, 300));
        } catch (IOException e) {
            System.err.println("Failed to load dataset: " + e.getMessage());
            System.exit(1);
        }
    }
}