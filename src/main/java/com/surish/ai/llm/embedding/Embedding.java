package com.surish.ai.llm.embedding;

public record Embedding(double[] values) {

    public int dimension() {
        return values.length;
    }

}