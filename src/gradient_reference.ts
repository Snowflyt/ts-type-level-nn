export type Weights = [number[][], number[][]];

export const copyWeights = (weights: Weights): Weights =>
  weights.map((matrix) => matrix.map((row) => [...row])) as Weights;

// This loss is deliberately independent of NeuralNetwork.forward and train.
export const loss = (weights: Weights, inputs: number[], targets: number[]): number => {
  const sigmoid = (value: number): number => 1 / (1 + Math.exp(-value));
  const hidden = weights[0].map((row) =>
    sigmoid(row.reduce((sum, weight, i) => sum + weight * inputs[i], 0)),
  );
  return weights[1].reduce((sum, row, i) => {
    const output = sigmoid(row.reduce((total, weight, j) => total + weight * hidden[j], 0));
    return sum + (output - targets[i]) ** 2 / 2;
  }, 0);
};

export const numericalGradient = (
  weights: Weights,
  inputs: number[],
  targets: number[],
  layer: number,
  row: number,
  column: number,
): number => {
  const epsilon = 1e-5;
  const plus = copyWeights(weights);
  const minus = copyWeights(weights);
  plus[layer][row][column] += epsilon;
  minus[layer][row][column] -= epsilon;
  return (loss(plus, inputs, targets) - loss(minus, inputs, targets)) / (2 * epsilon);
};
