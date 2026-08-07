package com.surish.ai.llm.embedding;

public interface EmbeddingLayer {

    Embedding lookup(int tokenId);

    int vocabularySize();

    int embeddingDimension();

}