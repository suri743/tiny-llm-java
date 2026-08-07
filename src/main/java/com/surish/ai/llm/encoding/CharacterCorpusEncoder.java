package com.surish.ai.llm.encoding;

import com.surish.ai.llm.vocabulary.Vocabulary;

import java.util.List;

public class CharacterCorpusEncoder
        implements CorpusEncoder<Character> {

    @Override
    public EncodedCorpus encode(List<Character> tokens,
                                Vocabulary<Character> vocabulary) {

        List<Integer> tokenIds = vocabulary.encode(tokens);

        return new EncodedCorpus(tokenIds);
    }
}