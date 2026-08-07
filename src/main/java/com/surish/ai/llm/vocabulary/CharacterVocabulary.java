package com.surish.ai.llm.vocabulary;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public record CharacterVocabulary(Map<Character, Integer> tokenToId,Map<Integer, Character> idToToken) implements Vocabulary<Character> {

    public CharacterVocabulary(Map<Character, Integer> tokenToId,
                               Map<Integer, Character> idToToken) {
        this.tokenToId = Map.copyOf(tokenToId);
        this.idToToken = Map.copyOf(idToToken);
    }

    @Override
    public int encode(Character token) {
        Integer id = tokenToId.get(token);

        if(id == null){
            throw new IllegalArgumentException("Unknown token : " + token);
        }

        return id;
    }

    @Override
    public Character decode(int id) {
        Character token = idToToken.get(id);

        if(token == null){
            throw new IllegalArgumentException("Unknown id : " + id);
        }

        return token;
    }

    @Override
    public List<Integer> encode(List<Character> tokens) {

        return tokens
            .stream()
            .map(this::encode)
            .toList();
    }

    @Override
    public List<Character> decode(List<Integer> ids) {

        return ids
            .stream()
            .map(this::decode)
            .collect(Collectors.toList());
    }

    @Override
    public int size() {
        return tokenToId.size();
    }

    @Override
    public boolean contains(Character token) {
        return tokenToId.containsKey(token);
    }
}