package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Vector;

public class SoftmaxLayer {

    public Vector forward(Vector logits) {
        double max = logits.get(0);
        for (int i = 1; i < logits.size(); i++) {
            if (logits.get(i) > max) max = logits.get(i);
        }

        double sum = 0.0;
        Vector probs = new Vector(logits.size());
        for (int i = 0; i < logits.size(); i++) {
            double val = Math.exp(logits.get(i) - max);
            probs.set(i, val);
            sum += val;
        }

        for (int i = 0; i < probs.size(); i++) {
            probs.set(i, probs.get(i) / sum);
        }

        return probs;
    }
}
