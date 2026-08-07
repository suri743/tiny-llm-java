package com.surish.ai.llm.tokenizer;

import com.surish.ai.llm.corpus.Corpus;

import java.util.List;

public interface Tokenizer<T> {

    List<T> tokenize(Corpus corpus);

}