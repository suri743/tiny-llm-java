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
        int trainingExamples = 100;

        System.out.println("\n--- Training on first " + trainingExamples + " token pairs ---\n");

        for (int step = 0; step < trainingExamples; step++) {

            int inputTokenId = encodedCorpus.get(step);
            int targetTokenId = encodedCorpus.get(step + 1);

            char inputChar = vocabulary.decode(inputTokenId);
            char targetChar = vocabulary.decode(targetTokenId);

            // Forward pass
            Embedding embedding = embeddingLayer.lookup(inputTokenId);
            Vector logits = denseLayer.forward(embedding.vector());
            Vector probs = softmaxLayer.forward(logits);

            // Target: one-hot over vocab size
            Vector target = new Vector(vocabulary.size());
            target.set(targetTokenId, 1.0);

            double loss = lossFunction.calculate(probs, target);

            // Backward pass
            Vector outputGradient = lossFunction.gradient(probs, target);
            Vector embeddingGradient = denseLayer.backward(embedding.vector(), outputGradient, learningRate);

            // Update embedding
            Vector embeddingVector = embedding.vector();
            for (int i = 0; i < embeddingVector.size(); i++) {
                embeddingVector.set(i, embeddingVector.get(i) - learningRate * embeddingGradient.get(i));
            }

            if (step < 5 || step == trainingExamples - 1) {
                System.out.printf("Step %3d | '%c' -> '%c' | loss: %.6f%n",
                    step, inputChar, targetChar, loss);
            }
        }

        System.out.println("\n--- Done: " + trainingExamples + " training steps complete ---");
    }
}
