package com.surish.ai.llm.embedding;

import com.surish.ai.llm.tensor.Matrix;
import com.surish.ai.llm.tensor.Vector;

import java.util.Random;

public class PositionalEmbeddingLayer {

    private final Matrix embeddings;

    public PositionalEmbeddingLayer(int contextSize, int embeddingDim) {
        embeddings = new Matrix(contextSize, embeddingDim);
        Random random = new Random();
        for (int row = 0; row < contextSize; row++) {
            for (int col = 0; col < embeddingDim; col++) {
                embeddings.set(row, col, random.nextGaussian() * 0.02);
            }
        }
    }

    public Vector lookup(int position) {
        return embeddings.row(position);
    }

    public void update(int position, Vector gradient, double learningRate) {
        Vector row = embeddings.row(position);
        for (int i = 0; i < row.size(); i++) {
            row.set(i, row.get(i) - learningRate * gradient.get(i));
        }
    }
}
