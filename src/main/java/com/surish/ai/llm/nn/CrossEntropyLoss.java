package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Vector;

public class CrossEntropyLoss implements LossFunction {

    @Override
    public double calculate(Vector probs, Vector target) {
        double loss = 0.0;
        for (int i = 0; i < probs.size(); i++) {
            if (target.get(i) == 1.0) {
                // -log(probability of correct token)
                loss = -Math.log(Math.max(probs.get(i), 1e-12));
                break;
            }
        }
        return loss;
    }

    @Override
    public Vector gradient(Vector probs, Vector target) {
        // gradient of softmax + cross-entropy combined = probs - one_hot
        Vector grad = new Vector(probs.size());
        for (int i = 0; i < probs.size(); i++) {
            grad.set(i, probs.get(i) - target.get(i));
        }
        return grad;
    }
}
