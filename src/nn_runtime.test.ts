import assert from "node:assert/strict";
import { test } from "node:test";

import { copyWeights, loss, numericalGradient } from "./gradient_reference.ts";
import { NeuralNetwork } from "./nn_runtime.ts";

import type { Weights } from "./gradient_reference.ts";

const gradientCases: { name: string; inputs: number[]; targets: number[]; weights: Weights }[] = [
  {
    name: "exact zero pre-activations",
    inputs: [1],
    targets: [0],
    weights: [[[0], [0]], [[1, -1]]],
  },
  {
    name: "single output",
    inputs: [0.8],
    targets: [0.2],
    weights: [[[0.4]], [[0.7]]],
  },
  {
    name: "multiple inputs, hidden nodes, and outputs",
    inputs: [0.8, -0.3],
    targets: [0.2, 0.9],
    weights: [
      [
        [0.4, -0.2],
        [0.1, 0.6],
      ],
      [
        [0.7, -0.5],
        [-0.3, 0.8],
      ],
    ],
  },
  {
    name: "outputs with different sigmoid derivatives",
    inputs: [1],
    targets: [0.99, 0.47],
    weights: [[[0]], [[6], [1]]],
  },
];

for (const { inputs, name, targets, weights } of gradientCases) {
  test(`both layers follow the numerical squared-error gradient: ${name}`, () => {
    const learningRate = 0.01;
    const before = copyWeights(weights);
    const nn = new NeuralNetwork(
      inputs.length,
      weights[0].length,
      targets.length,
      learningRate,
      ...copyWeights(weights),
    );
    nn.train(inputs, targets);
    const after: Weights = [nn.weightsInputHidden, nn.weightsHiddenOutput];

    before.forEach((matrix, layer) => {
      matrix.forEach((row, i) => {
        row.forEach((weight, j) => {
          const gradient = numericalGradient(before, inputs, targets, layer, i, j);
          const actual = (weight - after[layer][i][j]) / learningRate;
          assert.ok(
            Math.abs(actual - gradient) < 1e-8,
            `layer ${layer}, weight [${i}][${j}]: update gradient ${actual}, numerical ${gradient}`,
          );
        });
      });
    });
  });
}

test("the multi-output hidden update decreases loss when output weights are held fixed", () => {
  const inputs = [1];
  const targets = [0.99, 0.47];
  const before: Weights = [[[0]], [[6], [1]]];
  const nn = new NeuralNetwork(1, 1, 2, 0.01, ...copyWeights(before));
  nn.train(inputs, targets);

  assert.ok(numericalGradient(before, inputs, targets, 0, 0, 0) > 0);
  assert.ok(nn.weightsInputHidden[0][0] < 0);
  assert.ok(
    loss([nn.weightsInputHidden, before[1]], inputs, targets) < loss(before, inputs, targets),
  );
});

test("a zero learning rate leaves both layers unchanged", () => {
  const weights: Weights = [[[0.4]], [[0.7]]];
  const nn = new NeuralNetwork(1, 1, 1, 0, ...copyWeights(weights));
  nn.train([0.8], [0.2]);
  assert.deepEqual([nn.weightsInputHidden, nn.weightsHiddenOutput], weights);
});
