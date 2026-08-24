package com.surish.ai.llm.model;

import com.surish.ai.llm.embedding.EmbeddingLayer;
import com.surish.ai.llm.embedding.PositionalEmbeddingLayer;
import com.surish.ai.llm.embedding.RandomEmbeddingLayer;
import com.surish.ai.llm.nn.CrossEntropyLoss;
import com.surish.ai.llm.nn.DenseLayer;
import com.surish.ai.llm.nn.LossFunction;
import com.surish.ai.llm.nn.SoftmaxLayer;
import com.surish.ai.llm.nn.TransformerBlock;
import com.surish.ai.llm.tensor.Vector;
import com.surish.ai.llm.training.TrainingConfig;
import com.surish.ai.llm.vocabulary.Vocabulary;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class LanguageModel {

    public final EmbeddingLayer embeddingLayer;
    public final PositionalEmbeddingLayer positionalEmbeddingLayer;
    public final TransformerBlock[] blocks;
    public final DenseLayer outputLayer;
    public final SoftmaxLayer softmaxLayer;
    public final LossFunction lossFunction;
    public final TrainingConfig config;

    // cached between forward and backward
    public Vector lastBlockOutput;

    public LanguageModel(int vocabSize, TrainingConfig config) {
        this.config = config;
        this.embeddingLayer = new RandomEmbeddingLayer(vocabSize, config.embeddingDim);
        this.positionalEmbeddingLayer = new PositionalEmbeddingLayer(config.contextSize, config.embeddingDim);
        this.blocks = new TransformerBlock[config.numLayers];
        for (int i = 0; i < config.numLayers; i++)
            blocks[i] = new TransformerBlock(config.embeddingDim, config.numHeads);
        this.outputLayer = new DenseLayer(config.embeddingDim, vocabSize);
        this.softmaxLayer = new SoftmaxLayer();
        this.lossFunction = new CrossEntropyLoss();
    }

    public Vector forward(Vector[] tokens) {
        Vector[] x = tokens;
        for (TransformerBlock block : blocks)
            x = block.forward(x);
        lastBlockOutput = x[x.length - 1];
        Vector logits = outputLayer.forward(lastBlockOutput);
        return softmaxLayer.forward(logits);
    }

    public String predict(String seed, int length, Vocabulary<Character> vocabulary) {
        return predict(seed, length, vocabulary, 0.8);
    }

    public String predict(String seed, int length, Vocabulary<Character> vocabulary, double temperature) {
        List<Integer> contextIds = new ArrayList<>();
        for (int i = 0; i < config.contextSize; i++) {
            int seedIndex = i - (config.contextSize - seed.length());
            if (seedIndex < 0) {
                contextIds.add(vocabulary.encode(seed.charAt(0)));
            } else {
                contextIds.add(vocabulary.encode(seed.charAt(seedIndex)));
            }
        }

        StringBuilder output = new StringBuilder(seed);
        Random random = new Random();

        for (int i = 0; i < length; i++) {
            Vector[] tokens = buildTokens(contextIds);
            Vector probs = forward(tokens);
            int nextTokenId = sample(probs, temperature, random);
            output.append(vocabulary.decode(nextTokenId));
            contextIds.remove(0);
            contextIds.add(nextTokenId);
        }

        return output.toString();
    }

    private int sample(Vector probs, double temperature, Random random) {
        double[] scaled = new double[probs.size()];
        double sum = 0.0;
        for (int i = 0; i < probs.size(); i++) {
            scaled[i] = Math.pow(probs.get(i), 1.0 / temperature);
            sum += scaled[i];
        }
        double r = random.nextDouble() * sum;
        double cumulative = 0.0;
        for (int i = 0; i < scaled.length; i++) {
            cumulative += scaled[i];
            if (r <= cumulative) return i;
        }
        return scaled.length - 1;
    }

    private Vector[] buildTokens(List<Integer> contextIds) {
        Vector[] tokens = new Vector[config.contextSize];
        for (int c = 0; c < config.contextSize; c++) {
            Vector tokenEmb = embeddingLayer.lookup(contextIds.get(c));
            Vector posEmb = positionalEmbeddingLayer.lookup(c);
            Vector combined = new Vector(config.embeddingDim);
            for (int d = 0; d < config.embeddingDim; d++)
                combined.set(d, tokenEmb.get(d) + posEmb.get(d));
            tokens[c] = combined;
        }
        return tokens;
    }
}
