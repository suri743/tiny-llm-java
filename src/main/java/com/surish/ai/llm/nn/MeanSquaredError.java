package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Vector;

public class MeanSquaredError implements LossFunction {

    @Override
    public double calculate(Vector prediction, Vector target) {

        if (prediction.size() != target.size()) {
            throw new IllegalArgumentException(
                "Prediction and target must have the same size"
            );
        }

        double sum = 0.0;

        for (int i = 0; i < prediction.size(); i++) {
            double difference = prediction.get(i) - target.get(i);
            sum += difference * difference;
        }

        return sum / prediction.size();
    }

    @Override
    public Vector gradient(Vector prediction, Vector target) {

        Vector gradient =
            new Vector(prediction.size());

        double n = prediction.size();

        for (int i = 0; i < prediction.size(); i++) {

            double error =
                prediction.get(i) - target.get(i);

            gradient.set(i,(2.0 / n) * error);
        }

        return gradient;
    }
}