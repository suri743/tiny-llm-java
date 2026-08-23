package com.surish.ai;

import com.surish.ai.llm.corpus.ClasspathCorpusLoader;
import com.surish.ai.llm.corpus.Corpus;
import com.surish.ai.llm.corpus.CorpusLoader;
import com.surish.ai.llm.embedding.Embedding;
import com.surish.ai.llm.embedding.EmbeddingLayer;
import com.surish.ai.llm.embedding.RandomEmbeddingLayer;
import com.surish.ai.llm.encoding.CharacterCorpusEncoder;
import com.surish.ai.llm.encoding.CorpusEncoder;
import com.surish.ai.llm.encoding.EncodedCorpus;
import com.surish.ai.llm.nn.CrossEntropyLoss;
import com.surish.ai.llm.nn.DenseLayer;
import com.surish.ai.llm.nn.LossFunction;
import com.surish.ai.llm.nn.SoftmaxLayer;
import com.surish.ai.llm.tensor.Vector;
import com.surish.ai.llm.tokenizer.CharacterTokenizer;
import com.surish.ai.llm.tokenizer.Tokenizer;
import com.surish.ai.llm.vocabulary.CharacterVocabularyBuilder;
import com.surish.ai.llm.vocabulary.Vocabulary;
import com.surish.ai.llm.vocabulary.VocabularyBuilder;

import java.util.List;

public class Main {

    public static void main(String[] args) {

        // --- Setup ---

        CorpusLoader loader = new ClasspathCorpusLoader("tiny_shakespeare.txt");
        Corpus corpus = loader.load();

        Tokenizer<Character> tokenizer = new CharacterTokenizer();
        List<Character> tokens = tokenizer.tokenize(corpus);

        VocabularyBuilder<Character> builder = new CharacterVocabularyBuilder();
        Vocabulary<Character> vocabulary = builder.build(tokens);

        System.out.println("Tokens: " + tokens.size());
        System.out.println("Vocabulary size: " + vocabulary.size());

        CorpusEncoder<Character> encoder = new CharacterCorpusEncoder();
        EncodedCorpus encodedCorpus = encoder.encode(tokens, vocabulary);

        EmbeddingLayer embeddingLayer = new RandomEmbeddingLayer(vocabulary.size(), 16);
        DenseLayer denseLayer = new DenseLayer(16, vocabulary.size());
        SoftmaxLayer softmaxLayer = new SoftmaxLayer();
        LossFunction lossFunction = new CrossEntropyLoss();

        double learningRate = 0.01;
        int epochs = 20;
        int logInterval = 10000;
        int totalTokens = encodedCorpus.size() - 1;

        int trainSize = (int) (totalTokens * 0.9);
        int valSize = totalTokens - trainSize;

        System.out.println("Train tokens: " + trainSize);
        System.out.println("Val tokens:   " + valSize);
        System.out.println("\n--- Training ---\n");

        for (int epoch = 1; epoch <= epochs; epoch++) {

            // --- Training ---
            double epochLoss = 0.0;
            double intervalLoss = 0.0;

            for (int step = 0; step < trainSize; step++) {

                int inputTokenId = encodedCorpus.get(step);
                int targetTokenId = encodedCorpus.get(step + 1);

                Embedding embedding = embeddingLayer.lookup(inputTokenId);
                Vector logits = denseLayer.forward(embedding.vector());
                Vector probs = softmaxLayer.forward(logits);

                Vector target = new Vector(vocabulary.size());
                target.set(targetTokenId, 1.0);

                double loss = lossFunction.calculate(probs, target);
                epochLoss += loss;
                intervalLoss += loss;

                Vector outputGradient = lossFunction.gradient(probs, target);
                Vector embeddingGradient = denseLayer.backward(embedding.vector(), outputGradient, learningRate);

                Vector embeddingVector = embedding.vector();
                for (int i = 0; i < embeddingVector.size(); i++) {
                    embeddingVector.set(i, embeddingVector.get(i) - learningRate * embeddingGradient.get(i));
                }

                if ((step + 1) % logInterval == 0) {
                    System.out.printf("Epoch %d | Step %7d | avg loss: %.4f%n",
                        epoch, step + 1, intervalLoss / logInterval);
                    intervalLoss = 0.0;
                }
            }

            double trainAvgLoss = epochLoss / trainSize;

            // --- Validation (forward pass only, no weight updates) ---
            double valLoss = 0.0;
            for (int step = trainSize; step < totalTokens; step++) {
                int inputTokenId = encodedCorpus.get(step);
                int targetTokenId = encodedCorpus.get(step + 1);

                Embedding embedding = embeddingLayer.lookup(inputTokenId);
                Vector logits = denseLayer.forward(embedding.vector());
                Vector probs = softmaxLayer.forward(logits);

                Vector target = new Vector(vocabulary.size());
                target.set(targetTokenId, 1.0);

                valLoss += lossFunction.calculate(probs, target);
            }

            double valAvgLoss = valLoss / valSize;

            System.out.printf("%nEpoch %d complete | train loss: %.4f | val loss: %.4f%n%n",
                epoch, trainAvgLoss, valAvgLoss);
        }

        System.out.println("--- Training complete ---");
    }
}
