package com.surish.ai.llm.training;

import com.surish.ai.llm.encoding.EncodedCorpus;
import com.surish.ai.llm.model.LanguageModel;
import com.surish.ai.llm.tensor.Vector;

import java.time.LocalTime;
import java.time.format.DateTimeFormatter;

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
            System.out.printf("Epoch %d/%d%n", epoch, model.config.epochs);
            double trainLoss = runEpoch(true, epoch);
            double valLoss = runEpoch(false, epoch);
            System.out.printf("%nEpoch %d complete | train loss: %.4f | val loss: %.4f%n%n",
                epoch, trainLoss, valLoss);
        }

        System.out.println("--- Training complete ---");
    }

    private double runEpoch(boolean training, int epoch) {
        int start = training ? model.config.contextSize : trainSize;
        int end = training ? trainSize : totalTokens - 1;
        int total = end - start;
        double totalLoss = 0.0;

        long epochStart = System.currentTimeMillis();
        int logInterval = total / 20; // print 20 updates per epoch
        if (logInterval < 1) logInterval = 1;

        for (int step = start; step < end; step++) {
            Vector[] tokens = buildTokens(step);
            int targetTokenId = encodedCorpus.get(step + 1);

            Vector probs = model.forward(tokens);

            Vector target = new Vector(probs.size());
            target.set(targetTokenId, 1.0);

            totalLoss += model.lossFunction.calculate(probs, target);

            if (training) {
                Vector outputGradient = model.lossFunction.gradient(probs, target);
                outputGradient = clip(outputGradient, 1.0);
                Vector grad = model.outputLayer.backward(model.lastBlockOutput, outputGradient, model.config.learningRate);
                for (int b = model.blocks.length - 1; b >= 0; b--) {
                    grad = clip(grad, 1.0);
                    grad = model.blocks[b].backward(grad, model.config.learningRate);
                }
            }

            int done = step - start + 1;
            if (done % logInterval == 0 || done == total) {
                int pct = done * 100 / total;
                long elapsed = System.currentTimeMillis() - epochStart;
                long etaMs = done < total ? (elapsed * (total - done) / done) : 0;
                double avgLoss = totalLoss / done;
                String phase = training ? "train" : "val";
                System.out.printf("\r  [%s] %3d%% | loss: %.4f | elapsed: %s | eta: %s",
                    phase, pct, avgLoss, formatTime(elapsed), formatTime(etaMs));
            }
        }

        return totalLoss / total;
    }

    private String formatTime(long ms) {
        long secs = ms / 1000;
        if (secs < 60) return secs + "s";
        long mins = secs / 60;
        secs = secs % 60;
        if (mins < 60) return mins + "m " + secs + "s";
        long hrs = mins / 60;
        mins = mins % 60;
        return hrs + "h " + mins + "m";
    }

    private Vector clip(Vector v, double maxNorm) {
        double norm = 0.0;
        for (int i = 0; i < v.size(); i++) norm += v.get(i) * v.get(i);
        norm = Math.sqrt(norm);
        if (norm > maxNorm) {
            double scale = maxNorm / norm;
            Vector clipped = new Vector(v.size());
            for (int i = 0; i < v.size(); i++) clipped.set(i, v.get(i) * scale);
            return clipped;
        }
        return v;
    }

    private Vector[] buildTokens(int step) {
        int contextSize = model.config.contextSize;
        int embeddingDim = model.config.embeddingDim;

        Vector[] tokens = new Vector[contextSize];
        for (int c = 0; c < contextSize; c++) {
            int tokenId = encodedCorpus.get(step - contextSize + c);
            Vector tokenEmb = model.embeddingLayer.lookup(tokenId);
            Vector posEmb = model.positionalEmbeddingLayer.lookup(c);
            Vector combined = new Vector(embeddingDim);
            for (int d = 0; d < embeddingDim; d++)
                combined.set(d, tokenEmb.get(d) + posEmb.get(d));
            tokens[c] = combined;
        }
        return tokens;
    }
}
