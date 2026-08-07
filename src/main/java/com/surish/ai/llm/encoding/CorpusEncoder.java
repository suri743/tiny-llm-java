package com.surish.ai.llm.encoding;

import com.surish.ai.llm.vocabulary.Vocabulary;

import java.util.List;

public interface CorpusEncoder<T> {

    EncodedCorpus encode(List<T> tokens,
                         Vocabulary<T> vocabulary);

}