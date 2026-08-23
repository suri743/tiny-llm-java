package com.surish.ai.llm.embedding;

import com.surish.ai.llm.tensor.Vector;

public interface EmbeddingLayer {

    Vector lookup(int tokenId);

    int vocabularySize();

    int embeddingDimension();

}