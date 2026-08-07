package com.surish.ai.llm.corpus;

public record Corpus(String text) {

    public int length() {
        return text.length();
    }

    public boolean isEmpty() {
        return text.isEmpty();
    }

}