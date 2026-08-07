package com.surish.ai;

import com.surish.ai.llm.corpus.ClasspathCorpusLoader;
import com.surish.ai.llm.corpus.Corpus;
import com.surish.ai.llm.corpus.CorpusLoader;
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

        System.out.println("Corpus Size      : " + corpus.length());
        System.out.println("Token Count      : " + tokens.size());
        System.out.println("Vocabulary Size  : " + vocabulary.size());

        System.out.println();

        System.out.println("Z -> " + vocabulary.encode('Z'));
        System.out.println("20 -> " + vocabulary.decode(20));
        System.out.println(vocabulary.decode(vocabulary.encode('H')));

    }
}