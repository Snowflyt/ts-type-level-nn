const lcg = (seed: number): { value: number; nextSeed: number } => {
  const a = 1664525;
  const c = 1013904223;
  const m = 2 ** 32;
  return {
    value: ((a * seed + c) % m) / m, // Convert to [0, 1) range
    nextSeed: (a * seed + c) % m,
  };
};

const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));

const sigmoidDerivative = (x: number): number => x * (1 - x);

const randomMatrix = (rows: number, cols: number): number[][] => {
  let seed = 42;
  const next = () => {
    const { nextSeed, value } = lcg(seed);
    seed = nextSeed;
    return value;
  };
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => next() - 0.5));
};

export class NeuralNetwork {
  public learningRate: number;
  public weightsInputHidden: number[][];
  public weightsHiddenOutput: number[][];

  constructor(
    inputNodes: number,
    hiddenNodes: number,
    outputNodes: number,
    learningRate: number,
    weightsInputHidden: number[][] = randomMatrix(hiddenNodes, inputNodes),
    weightsHiddenOutput: number[][] = randomMatrix(outputNodes, hiddenNodes),
  ) {
    this.learningRate = learningRate;
    this.weightsInputHidden = weightsInputHidden;
    this.weightsHiddenOutput = weightsHiddenOutput;
  }

  forward = (inputArray: number[]): { hiddenOutputs: number[]; finalOutputs: number[] } => {
    const hiddenInputs = this.weightsInputHidden.map((row) =>
      row.reduce((sum, weight, i) => sum + weight * inputArray[i], 0),
    );
    const hiddenOutputs = hiddenInputs.map(sigmoid);

    const finalInputs = this.weightsHiddenOutput.map((row) =>
      row.reduce((sum, weight, i) => sum + weight * hiddenOutputs[i], 0),
    );
    const finalOutputs = finalInputs.map(sigmoid);

    return { hiddenOutputs, finalOutputs };
  };

  train = (inputArray: number[], targetArray: number[]) => {
    const { finalOutputs, hiddenOutputs } = this.forward(inputArray);

    const outputErrors = targetArray.map((target, i) => target - finalOutputs[i]);
    // Include the output sigmoid derivative before crossing into the hidden layer.
    // These deltas are negative pre-activation gradients of the half-squared-error loss.
    const outputDeltas = outputErrors.map((error, i) => error * sigmoidDerivative(finalOutputs[i]));

    const hiddenErrors = this.weightsHiddenOutput[0].map((_, j) =>
      this.weightsHiddenOutput.reduce((sum, row, i) => sum + row[j] * outputDeltas[i], 0),
    );

    this.weightsHiddenOutput = NeuralNetwork.updateWeights(
      this.weightsHiddenOutput,
      this.learningRate,
      outputErrors,
      finalOutputs,
      hiddenOutputs,
    );
    this.weightsInputHidden = NeuralNetwork.updateWeights(
      this.weightsInputHidden,
      this.learningRate,
      hiddenErrors,
      hiddenOutputs,
      inputArray,
    );
  };

  private static updateWeights = (
    weights: number[][],
    learningRate: number,
    errors: number[],
    outputs: number[],
    inputs: number[],
  ): number[][] => {
    const updateWeight = (weight: number, error: number, output: number, input: number): number =>
      weight + learningRate * error * sigmoidDerivative(output) * input;

    const updateRow = (row: number[], rowIndex: number, colIndex: number = 0): number[] => {
      if (colIndex >= row.length) return [];
      const updatedWeight = updateWeight(
        row[colIndex],
        errors[rowIndex],
        outputs[rowIndex],
        inputs[colIndex],
      );
      return [updatedWeight, ...updateRow(row, rowIndex, colIndex + 1)];
    };

    const updateMatrix = (matrix: number[][], rowIndex: number = 0): number[][] => {
      if (rowIndex >= matrix.length) return [];
      const updatedRow = updateRow(matrix[rowIndex], rowIndex);
      return [updatedRow, ...updateMatrix(matrix, rowIndex + 1)];
    };

    return updateMatrix(weights);
  };

  predict = (inputArray: number[]): number[] => {
    const { finalOutputs } = this.forward(inputArray);
    return finalOutputs;
  };
}
