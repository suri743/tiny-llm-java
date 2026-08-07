package com.surish.ai.llm.corpus;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

public class ClasspathCorpusLoader implements CorpusLoader {

    private final String resourceName;

    public ClasspathCorpusLoader(String resourceName) {
        this.resourceName = resourceName;
    }

    @Override
    public Corpus load() {

        try (InputStream stream = getClass()
                .getClassLoader()
                .getResourceAsStream(resourceName)) {

            if (stream == null) {
                throw new IllegalArgumentException(
                        "Resource not found: " + resourceName);
            }

            String text = new String(
                    stream.readAllBytes(),
                    StandardCharsets.UTF_8
            );

            return new Corpus(text);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

}