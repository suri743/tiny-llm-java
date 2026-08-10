package com.surish.ai.llm.nn;

import com.surish.ai.llm.tensor.Vector;

public interface LossFunction {

    double calculate(Vector prediction, Vector target);

    Vector gradient(Vector prediction, Vector target);
}