package com.surish.ai.llm.vocabulary;

import java.util.Collection;

public interface VocabularyBuilder<T> {

    Vocabulary<T> build(Collection<T> tokens);

}