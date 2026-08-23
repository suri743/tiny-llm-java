package com.surish.ai.llm.training;

public class TrainingConfig {

    public final int contextSize;
    public final int embeddingDim;
    public final int epochs;
    public final double learningRate;
    public final double trainSplit;

    public TrainingConfig(int contextSize, int embeddingDim, int epochs, double learningRate, double trainSplit) {
        this.contextSize = contextSize;
        this.embeddingDim = embeddingDim;
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.trainSplit = trainSplit;
    }
}
