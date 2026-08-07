package com.surish.ai.llm.vocabulary;

import java.util.List;

public interface Vocabulary<T> {

    int encode(T token);

    T decode(int id);

    List<Integer> encode(List<T> tokens);

    List<T> decode(List<Integer> ids);

    int size();

    boolean contains(T token);

}