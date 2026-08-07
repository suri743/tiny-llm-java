package com.surish.ai.llm.tokenizer;

import com.surish.ai.llm.corpus.Corpus;

import java.util.List;

public class CharacterTokenizer implements Tokenizer<Character> {

    @Override
    public List<Character> tokenize(Corpus corpus) {

        return corpus.text()
                .chars()
                .mapToObj(c -> (char) c)
                .toList();
    }

}