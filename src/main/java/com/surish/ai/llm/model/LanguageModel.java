package com.surish.ai.llm.model;

import com.surish.ai.llm.embedding.EmbeddingLayer;
import com.surish.ai.llm.embedding.RandomEmbeddingLayer;
import com.surish.ai.llm.nn.CrossEntropyLoss;
import com.surish.ai.llm.nn.DenseLayer;
import com.surish.ai.llm.nn.LossFunction;
import com.surish.ai.llm.nn.SoftmaxLayer;
import com.surish.ai.llm.tensor.Vector;
import com.surish.ai.llm.training.TrainingConfig;
import com.surish.ai.llm.vocabulary.Vocabulary;

import java.util.ArrayList;
import java.util.List;

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

    public String predict(String seed, int length, Vocabulary<Character> vocabulary) {
        // build initial context from seed characters, pad with first token if seed is short
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
            // build context vector from current window
            Vector context = new Vector(config.contextSize * config.embeddingDim);
            for (int c = 0; c < config.contextSize; c++) {
                Vector emb = embeddingLayer.lookup(contextIds.get(c)).vector();
                for (int d = 0; d < config.embeddingDim; d++) {
                    context.set(c * config.embeddingDim + d, emb.get(d));
                }
            }

            // forward pass → probabilities
            Vector probs = forward(context);

            // pick highest probability token
            int nextTokenId = 0;
            double maxProb = probs.get(0);
            for (int j = 1; j < probs.size(); j++) {
                if (probs.get(j) > maxProb) {
                    maxProb = probs.get(j);
                    nextTokenId = j;
                }
            }

            output.append(vocabulary.decode(nextTokenId));

            // slide context window
            contextIds.remove(0);
            contextIds.add(nextTokenId);
        }

        return output.toString();
    }
}
