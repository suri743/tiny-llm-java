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
import com.surish.ai.llm.nn.DenseLayer;
import com.surish.ai.llm.nn.LinearNeuron;
import com.surish.ai.llm.nn.LossFunction;
import com.surish.ai.llm.nn.MeanSquaredError;
import com.surish.ai.llm.nn.Neuron;
import com.surish.ai.llm.tensor.Vector;
import com.surish.ai.llm.tokenizer.CharacterTokenizer;
import com.surish.ai.llm.tokenizer.Tokenizer;
import com.surish.ai.llm.vocabulary.CharacterVocabularyBuilder;
import com.surish.ai.llm.vocabulary.Vocabulary;
import com.surish.ai.llm.vocabulary.VocabularyBuilder;

import java.util.List;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {

        CorpusLoader loader =
            new ClasspathCorpusLoader("tiny_shakespeare.txt");

        Corpus corpus = loader.load();

        Tokenizer<Character> tokenizer =
            new CharacterTokenizer();

        List<Character> tokens =
            tokenizer.tokenize(corpus);

        VocabularyBuilder<Character> builder =
            new CharacterVocabularyBuilder();

        Vocabulary<Character> vocabulary =
            builder.build(tokens);

        CorpusEncoder<Character> encoder =
            new CharacterCorpusEncoder();

        EncodedCorpus encodedCorpus =
            encoder.encode(tokens, vocabulary);

//        System.out.println(encodedCorpus.tokenIds().subList(0, 10));

        EmbeddingLayer embeddingLayer =
            new RandomEmbeddingLayer(
                vocabulary.size(),16);

        Embedding embedding =
            embeddingLayer.lookup(vocabulary.encode('T'));

//        Neuron neuron = new LinearNeuron(embedding.dimension());
//
//        double output = neuron.forward(embedding.vector());
//
//        System.out.println("----------------------------------------------------");
//
//        System.out.println("Embedding");
//        System.out.println(embedding.vector());
//
//        System.out.println("----------------------------------------------------");
//
//        System.out.println("Neuron Output");
//        System.out.println(output);

        DenseLayer denseLayer =
            new DenseLayer(
                embedding.dimension(),
                8);

        Vector output =
            denseLayer.forward(embedding.vector());

// Target for this training example
        Vector target = new Vector(8);

// For now, we are manually defining the target
        target.set(0, 1.0);
        target.set(1, 0.0);
        target.set(2, 0.0);
        target.set(3, 0.0);
        target.set(4, 0.0);
        target.set(5, 0.0);
        target.set(6, 0.0);
        target.set(7, 0.0);

        LossFunction lossFunction =
            new MeanSquaredError();

        double loss =
            lossFunction.calculate(output, target);

        Vector gradient =
            lossFunction.gradient(output, target);

        System.out.println();

        System.out.println("Embedding");
        System.out.println(embedding.vector());

        System.out.println("----------------------------------------------------");

        System.out.println("Dense Layer Output");
        System.out.println(output);

        System.out.println("----------------------------------------------------");

        System.out.println("Target");
        System.out.println(target);

        System.out.println("----------------------------------------------------");

        System.out.println("Loss");
        System.out.println(loss);

        System.out.println("----------------------------------------------------");

        System.out.println("Loss Gradient");
        System.out.println(gradient);

    }
}