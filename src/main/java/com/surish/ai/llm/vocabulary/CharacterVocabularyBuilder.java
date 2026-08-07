package com.surish.ai.llm.vocabulary;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeSet;

public class CharacterVocabularyBuilder
        implements VocabularyBuilder<Character> {

    @Override
    public Vocabulary<Character> build(Collection<Character> tokens) {

        TreeSet<Character> uniqueTokens = new TreeSet<>(tokens);

        Map<Character, Integer> tokenToId = new LinkedHashMap<>();
        Map<Integer, Character> idToToken = new LinkedHashMap<>();

        int id = 0;

        for (Character token : uniqueTokens) {

            tokenToId.put(token, id);
            idToToken.put(id, token);

            id++;
        }

        return new CharacterVocabulary(tokenToId, idToToken);
    }
}