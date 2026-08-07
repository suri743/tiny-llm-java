package com.surish.ai.llm.embedding;

import com.surish.ai.llm.tensor.Vector;

public record Embedding(Vector vector) {

    public int dimension() {
        return vector.size();
    }

}