import { NeuralNetwork } from "./nn_runtime.ts";

const main = () => {
  const nn = new NeuralNetwork(1, 4, 1, 10);
  const trainData = [
    { input: [1], target: [0.1] },
    { input: [2], target: [0.2] },
  ];
  const testData = [{ input: [3], target: [0.3] }];

  for (let i = 0; i < 10; i++)
    trainData.forEach((data) => {
      nn.train(data.input, data.target);
    });

  console.log("Training complete");
  console.log("Weights Input Hidden: ", nn.weightsInputHidden);
  console.log("Weights Hidden Output: ", nn.weightsHiddenOutput);

  const stringifyArray = (arr: unknown[]): string =>
    "[" + arr.map((v) => String(v)).join(", ") + "]";

  for (const data of [...trainData, ...testData]) {
    const output = nn.predict(data.input);
    console.log(
      `Input: ${stringifyArray(data.input)} => Predicted Output: ${stringifyArray(output.map((v) => v.toFixed(2)))}`,
    );
  }
};

main();
