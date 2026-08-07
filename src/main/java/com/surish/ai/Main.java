package com.surish.ai;

import com.surish.ai.llm.corpus.ClasspathCorpusLoader;
import com.surish.ai.llm.corpus.Corpus;
import com.surish.ai.llm.corpus.CorpusLoader;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {

        CorpusLoader loader =
            new ClasspathCorpusLoader("tiny_shakespeare.txt");

        Corpus corpus = loader.load();

        System.out.println("Characters : " + corpus.length());

        System.out.println();

        System.out.println(corpus.text().substring(0,300));

    }
}