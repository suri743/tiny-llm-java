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

        System.out.println(encodedCorpus.tokenIds().subList(0, 10));

        EmbeddingLayer embeddingLayer =
            new RandomEmbeddingLayer(
                vocabulary.size(),
                16);

        Embedding embedding =
            embeddingLayer.lookup(
                vocabulary.encode('H'));

        System.out.println("-------------------------------------------------");

        System.out.println(
            embedding.vector()
        );

    }
}