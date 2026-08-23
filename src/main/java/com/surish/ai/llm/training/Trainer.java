package com.surish.ai.llm.training;

import com.surish.ai.llm.encoding.EncodedCorpus;
import com.surish.ai.llm.model.LanguageModel;
import com.surish.ai.llm.tensor.Vector;

public class Trainer {

    private final LanguageModel model;
    private final EncodedCorpus encodedCorpus;
    private final int trainSize;
    private final int totalTokens;

    public Trainer(LanguageModel model, EncodedCorpus encodedCorpus) {
        this.model = model;
        this.encodedCorpus = encodedCorpus;
        this.totalTokens = encodedCorpus.size() - 1;
        this.trainSize = (int) (totalTokens * model.config.trainSplit);
    }

    public void train() {
        int valSize = totalTokens - trainSize;
        System.out.println("Train tokens: " + trainSize);
        System.out.println("Val tokens:   " + valSize);
        System.out.println("\n--- Training ---\n");

        for (int epoch = 1; epoch <= model.config.epochs; epoch++) {
            double trainLoss = runEpoch(epoch, true);
            double valLoss = runEpoch(epoch, false);
            System.out.printf("Epoch %d complete | train loss: %.4f | val loss: %.4f%n%n",
                epoch, trainLoss, valLoss);
        }

        System.out.println("--- Training complete ---");
    }

    private double runEpoch(int epoch, boolean training) {
        int start = training ? model.config.contextSize : trainSize;
        int end = training ? trainSize : totalTokens;
        double totalLoss = 0.0;

        for (int step = start; step < end; step++) {
            Vector context = buildContext(step);
            int targetTokenId = encodedCorpus.get(step + 1);

            Vector probs = model.forward(context);

            Vector target = new Vector(probs.size());
            target.set(targetTokenId, 1.0);

            totalLoss += model.lossFunction.calculate(probs, target);

            if (training) {
                Vector outputGradient = model.lossFunction.gradient(probs, target);
                model.denseLayer.backward(context, outputGradient, model.config.learningRate);
            }
        }

        return totalLoss / (end - start);
    }

    private Vector buildContext(int step) {
        int contextSize = model.config.contextSize;
        int embeddingDim = model.config.embeddingDim;

        Vector context = new Vector(contextSize * embeddingDim);
        for (int c = 0; c < contextSize; c++) {
            int tokenId = encodedCorpus.get(step - contextSize + c);
            Vector tokenEmb = model.embeddingLayer.lookup(tokenId);
            Vector posEmb = model.positionalEmbeddingLayer.lookup(c);
            for (int d = 0; d < embeddingDim; d++) {
                context.set(c * embeddingDim + d, tokenEmb.get(d) + posEmb.get(d));
            }
        }
        return context;
    }
}
