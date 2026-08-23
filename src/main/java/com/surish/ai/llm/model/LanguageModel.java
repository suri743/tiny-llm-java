package com.surish.ai.llm.model;

import com.surish.ai.llm.embedding.EmbeddingLayer;
import com.surish.ai.llm.embedding.RandomEmbeddingLayer;
import com.surish.ai.llm.nn.CrossEntropyLoss;
import com.surish.ai.llm.nn.DenseLayer;
import com.surish.ai.llm.nn.LossFunction;
import com.surish.ai.llm.nn.SoftmaxLayer;
import com.surish.ai.llm.tensor.Vector;
import com.surish.ai.llm.training.TrainingConfig;

public class LanguageModel {

    public final EmbeddingLayer embeddingLayer;
    public final DenseLayer denseLayer;
    public final SoftmaxLayer softmaxLayer;
    public final LossFunction lossFunction;
    public final TrainingConfig config;

    public LanguageModel(int vocabSize, TrainingConfig config) {
        this.config = config;
        this.embeddingLayer = new RandomEmbeddingLayer(vocabSize, config.embeddingDim);
        this.denseLayer = new DenseLayer(config.contextSize * config.embeddingDim, vocabSize);
        this.softmaxLayer = new SoftmaxLayer();
        this.lossFunction = new CrossEntropyLoss();
    }

    public Vector forward(Vector context) {
        Vector logits = denseLayer.forward(context);
        return softmaxLayer.forward(logits);
    }
}
