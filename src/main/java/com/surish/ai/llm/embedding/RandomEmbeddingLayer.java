package com.surish.ai.llm.embedding;

import com.surish.ai.llm.tensor.Matrix;

import java.util.Random;

public class RandomEmbeddingLayer implements EmbeddingLayer {

    private final Matrix embeddings;

    private final Random random = new Random();

    public RandomEmbeddingLayer(int vocabularySize,
                                int embeddingDimension) {

        embeddings = new Matrix(vocabularySize, embeddingDimension);

        initialize();
    }

    private void initialize() {

        for (int row = 0; row < embeddings.rows(); row++) {

            for (int column = 0; column < embeddings.columns(); column++) {

                embeddings.set(
                        row,
                        column,
                        random.nextGaussian() * 0.02
                );
            }
        }
    }

    @Override
    public Embedding lookup(int tokenId) {

        return new Embedding(
                embeddings.row(tokenId)
        );
    }

    @Override
    public int vocabularySize() {
        return embeddings.rows();
    }

    @Override
    public int embeddingDimension() {
        return embeddings.columns();
    }

}