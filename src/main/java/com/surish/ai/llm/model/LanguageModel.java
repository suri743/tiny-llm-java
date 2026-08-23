package com.surish.ai.llm.model;

import com.surish.ai.llm.embedding.EmbeddingLayer;
import com.surish.ai.llm.embedding.PositionalEmbeddingLayer;
import com.surish.ai.llm.embedding.RandomEmbeddingLayer;
import com.surish.ai.llm.nn.CrossEntropyLoss;
import com.surish.ai.llm.nn.DenseLayer;
import com.surish.ai.llm.nn.LossFunction;
import com.surish.ai.llm.nn.SelfAttention;
import com.surish.ai.llm.nn.SoftmaxLayer;
import com.surish.ai.llm.tensor.Vector;
import com.surish.ai.llm.training.TrainingConfig;
import com.surish.ai.llm.vocabulary.Vocabulary;

import java.util.ArrayList;
import java.util.List;

public class LanguageModel {

    public final EmbeddingLayer embeddingLayer;
    public final PositionalEmbeddingLayer positionalEmbeddingLayer;
    public final SelfAttention selfAttention;
    public final DenseLayer outputLayer;
    public final SoftmaxLayer softmaxLayer;
    public final LossFunction lossFunction;
    public final TrainingConfig config;

    public LanguageModel(int vocabSize, TrainingConfig config) {
        this.config = config;
        this.embeddingLayer = new RandomEmbeddingLayer(vocabSize, config.embeddingDim);
        this.positionalEmbeddingLayer = new PositionalEmbeddingLayer(config.contextSize, config.embeddingDim);
        this.selfAttention = new SelfAttention(config.embeddingDim);
        this.outputLayer = new DenseLayer(config.embeddingDim, vocabSize);
        this.softmaxLayer = new SoftmaxLayer();
        this.lossFunction = new CrossEntropyLoss();
    }

    // forward pass takes 8 separate token vectors
    public Vector forward(Vector[] tokens) {
        Vector[] attended = selfAttention.forward(tokens);
        // use last token's attended vector for prediction
        Vector logits = outputLayer.forward(attended[attended.length - 1]);
        return softmaxLayer.forward(logits);
    }

    public String predict(String seed, int length, Vocabulary<Character> vocabulary) {
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

        for (int i = 0; i < length; i++) {
            Vector[] tokens = buildTokens(contextIds);
            Vector probs = forward(tokens);

            int nextTokenId = 0;
            double maxProb = probs.get(0);
            for (int j = 1; j < probs.size(); j++) {
                if (probs.get(j) > maxProb) {
                    maxProb = probs.get(j);
                    nextTokenId = j;
                }
            }

            output.append(vocabulary.decode(nextTokenId));
            contextIds.remove(0);
            contextIds.add(nextTokenId);
        }

        return output.toString();
    }

    private Vector[] buildTokens(List<Integer> contextIds) {
        Vector[] tokens = new Vector[config.contextSize];
        for (int c = 0; c < config.contextSize; c++) {
            Vector tokenEmb = embeddingLayer.lookup(contextIds.get(c));
            Vector posEmb = positionalEmbeddingLayer.lookup(c);
            Vector combined = new Vector(config.embeddingDim);
            for (int d = 0; d < config.embeddingDim; d++) {
                combined.set(d, tokenEmb.get(d) + posEmb.get(d));
            }
            tokens[c] = combined;
        }
        return tokens;
    }
}
