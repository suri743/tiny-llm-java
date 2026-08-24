package com.surish.ai.llm.training;

public class TrainingConfig {

    public final int contextSize;
    public final int embeddingDim;
    public final int numLayers;
    public final int numHeads;
    public final int epochs;
    public final double learningRate;
    public final double trainSplit;

    public TrainingConfig(int contextSize, int embeddingDim, int numLayers, int numHeads, int epochs, double learningRate, double trainSplit) {
        this.contextSize = contextSize;
        this.embeddingDim = embeddingDim;
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.epochs = epochs;
        this.learningRate = learningRate;
        this.trainSplit = trainSplit;
    }
}
