package com.surish.ai.dataset;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class DatasetLoader {

    public String load() throws IOException {

        try (InputStream inputStream = getClass()
            .getClassLoader()
            .getResourceAsStream("tiny_shakespeare.txt")) {

            if(inputStream == null){
                throw new IOException("Dataset resource not found: tiny_shakespeare.txt");
            }

            return new String(inputStream.readAllBytes(),StandardCharsets.UTF_8);
        }
    }
}