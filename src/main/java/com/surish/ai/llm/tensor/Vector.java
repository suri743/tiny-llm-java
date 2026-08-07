package com.surish.ai.llm.tensor;

import lombok.EqualsAndHashCode;
import lombok.Getter;

import java.util.Arrays;

@Getter
@EqualsAndHashCode
public class Vector {

    private final double[] values;

    public Vector(int size) {
        this.values = new double[size];
    }

    public Vector(double[] values) {
        this.values = values.clone();
    }

    public int size() {
        return values.length;
    }

    public double get(int index) {
        return values[index];
    }

    public void set(int index, double value) {
        values[index] = value;
    }

    public double[] toArray() {
        return values.clone();
    }

    @Override
    public String toString() {
        return Arrays.toString(values);
    }
}