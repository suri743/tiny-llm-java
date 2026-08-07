package com.surish.ai.llm.encoding;

import java.util.List;

public record EncodedCorpus(List<Integer> tokenIds) {

    public int size() {
        return tokenIds.size();
    }

    public int get(int index) {
        return tokenIds.get(index);
    }

}