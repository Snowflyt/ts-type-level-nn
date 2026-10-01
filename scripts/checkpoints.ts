import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { format, resolveConfig } from "prettier";
import * as ts from "typescript";

import { numericalGradient } from "../src/gradient_reference.ts";

import type { Weights } from "../src/gradient_reference.ts";
type SourceFile = ts.SourceFile;
type TypeAliasDeclaration = ts.TypeAliasDeclaration;
type Checker = ts.TypeChecker;
type Type = ts.Type;

const { values } = parseArgs({
  options: {
    check: { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
});
if (values.help) {
  console.log("Usage: node scripts/checkpoints.ts [--check]");
  process.exit(0);
}

const root = fileURLToPath(new URL("../", import.meta.url));
const mainFile = resolve(root, "src/main.ts");
const configFile = resolve(root, "tsconfig.json");
let source = await readFile(mainFile, "utf8");

type State = {
  LearningRate: number;
  WeightsInputHidden: number[][];
  WeightsHiddenOutput: number[][];
};
type Step = {
  index: number;
  pos: number;
  end: number;
  inputs: number[];
  targets: number[];
  saved: State;
};

const declaration = (file: SourceFile, name: string): TypeAliasDeclaration => {
  const alias = file.statements.find(
    (node): node is TypeAliasDeclaration =>
      ts.isTypeAliasDeclaration(node) && node.name.text === name,
  );
  assert.ok(alias, `Missing type alias ${name}`);
  return alias;
};
const describe = (checker: Checker, type: Type | undefined): string =>
  type ? checker.typeToString(type) : "missing type";
const tuple = (checker: Checker, type: Type | undefined, context: string): readonly Type[] => {
  assert.ok(
    type && checker.isTupleType(type),
    `${context}: expected a concrete tuple, got ${describe(checker, type)}`,
  );
  return checker.getTypeArguments(type as ts.TypeReference);
};
const number = (checker: Checker, type: Type | undefined, context: string): number => {
  assert.ok(
    type &&
      (type.flags & ts.TypeFlags.NumberLiteral) !== 0 &&
      Number.isFinite((type as ts.NumberLiteralType).value),
    `${context}: expected a numeric literal, got ${describe(checker, type)}`,
  );
  return (type as ts.NumberLiteralType).value;
};
const state = (
  checker: Checker,
  type: Type | undefined,
  node: TypeAliasDeclaration,
  context: string,
): State => {
  assert.ok(
    type && (type.flags & ts.TypeFlags.Any) === 0,
    `${context}: expected a concrete network state`,
  );
  const property = (name: string): Type | undefined => {
    const symbol = checker.getPropertyOfType(type, name);
    assert.ok(symbol, `${context}: missing property ${name}`);
    return checker.getTypeOfSymbolAtLocation(symbol, node);
  };
  const matrix = (name: string): number[][] =>
    tuple(checker, property(name), `${context}.${name}`).map((row, i) =>
      tuple(checker, row, `${context}.${name}[${i}]`).map((value, j) =>
        number(checker, value, `${context}.${name}[${i}][${j}]`),
      ),
    );
  return {
    LearningRate: number(checker, property("LearningRate"), `${context}.LearningRate`),
    WeightsInputHidden: matrix("WeightsInputHidden"),
    WeightsHiddenOutput: matrix("WeightsHiddenOutput"),
  };
};
const printState = (value: State): string =>
  `{
  LearningRate: ${value.LearningRate};
  WeightsInputHidden: ${JSON.stringify(value.WeightsInputHidden)};
  WeightsHiddenOutput: ${JSON.stringify(value.WeightsHiddenOutput)};
}`;

const validateGradient = (
  before: State,
  after: State,
  inputs: number[],
  targets: number[],
): number => {
  const weights: Weights = [before.WeightsInputHidden, before.WeightsHiddenOutput];
  const updated: Weights = [after.WeightsInputHidden, after.WeightsHiddenOutput];
  assert.ok(before.LearningRate > 0);
  assert.equal(after.LearningRate, before.LearningRate);
  let maximumError = 0;
  weights.forEach((matrix, layer) => {
    assert.equal(updated[layer].length, matrix.length);
    matrix.forEach((row, i) => {
      assert.equal(updated[layer][i].length, row.length);
      row.forEach((weight, j) => {
        const expected = numericalGradient(weights, inputs, targets, layer, i, j);
        const actual = (weight - updated[layer][i][j]) / before.LearningRate;
        maximumError = Math.max(maximumError, Math.abs(actual - expected));
      });
    });
  });
  // Exp uses a finite Taylor series at the type level, unlike the independent Math.exp loss.
  assert.ok(maximumError < 1e-6, `Numerical gradient mismatch: ${maximumError}`);
  return maximumError;
};

const steps: Step[] = [];
const config = ts.readConfigFile(configFile, ts.sys.readFile);
assert.equal(config.error, undefined);
const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, root);
assert.deepEqual(parsed.errors, []);
const parser = ts.createProgram([mainFile], parsed.options);
const checker = parser.getTypeChecker();
const file = parser.getSourceFile(mainFile);
assert.ok(file && file.text === source);
const prelude = source.slice(0, declaration(file, "NN").end);
const implementation = source.slice(declaration(file, "LCG").pos);
for (let i = 1; i <= 20; i++) {
  const step = declaration(file, `NextNN${i}`);
  const expected = declaration(file, `expected_nn${i}`);
  assert.ok(ts.isTypeReferenceNode(step.type) && step.type.typeArguments?.length === 3);
  const [startType, inputsType, targetsType] = step.type.typeArguments;
  assert.equal(
    source.slice(startType.pos, startType.end).trim(),
    i === 1 ? "NN" : `expected_nn${i - 1}`,
    `Checkpoint ${i} must start from the preceding literal state`,
  );
  const numbers = (type: Type | undefined, context: string): number[] =>
    tuple(checker, type, context).map((value, j) => number(checker, value, `${context}[${j}]`));
  steps.push({
    index: i,
    pos: expected.type.pos,
    end: expected.type.end,
    inputs: numbers(checker.getTypeFromTypeNode(inputsType), `Checkpoint ${i} inputs`),
    targets: numbers(checker.getTypeFromTypeNode(targetsType), `Checkpoint ${i} targets`),
    saved: state(checker, checker.getTypeAtLocation(expected), expected, `Checkpoint ${i} saved`),
  });
}

const stale: number[] = [];
const generated: { pos: number; end: number; value: State }[] = [];
let maximumError = 0;
let previous: State | undefined;
for (const { end, index, inputs, pos, saved, targets } of steps) {
  // Check a single transition using the implementation copied verbatim from main.ts.
  const query = `${prelude}
type CurrentNN = ${previous ? printState(previous) : "NN"};
type GeneratedNN = NN.Train<CurrentNN, ${JSON.stringify(inputs)}, ${JSON.stringify(targets)}>;
${implementation}`;
  const host = ts.createCompilerHost(parsed.options);
  const getSourceFile = host.getSourceFile;
  host.getSourceFile = (name, languageVersion, onError, fresh) =>
    resolve(name) === mainFile ?
      ts.createSourceFile(name, query, languageVersion, true)
    : getSourceFile(name, languageVersion, onError, fresh);
  const program = ts.createProgram([mainFile], parsed.options, host);
  const diagnostics = ts.getPreEmitDiagnostics(program);
  assert.equal(
    diagnostics.length,
    0,
    `Checkpoint ${index} compiler diagnostics: ${diagnostics.map((d) => ts.flattenDiagnosticMessageText(d.messageText, "\n")).join("\n")}`,
  );
  const checker = program.getTypeChecker();
  const file = program.getSourceFile(mainFile);
  assert.ok(file && file.text === query);
  const beforeNode = declaration(file, "CurrentNN");
  const actualNode = declaration(file, "GeneratedNN");
  const before = state(
    checker,
    checker.getTypeAtLocation(beforeNode),
    beforeNode,
    `Checkpoint ${index} before`,
  );
  const actual = state(
    checker,
    checker.getTypeAtLocation(actualNode),
    actualNode,
    `Checkpoint ${index} generated`,
  );
  const error = validateGradient(before, actual, inputs, targets);
  maximumError = Math.max(maximumError, error);
  if (JSON.stringify(actual) !== JSON.stringify(saved)) stale.push(index);
  generated.push({ pos, end, value: actual });
  previous = actual;
  console.log(`Checkpoint ${index}: numerical gradient error ${error}`);
}

if (values.check) {
  assert.deepEqual(stale, [], `Stale checkpoints: ${stale.join(", ")}`);
} else {
  for (const { end, pos, value } of generated.reverse())
    source = source.slice(0, pos) + " " + printState(value) + source.slice(end);
  await writeFile(
    mainFile,
    await format(source, { ...(await resolveConfig(mainFile)), filepath: mainFile }),
  );
}
console.log(`Verified 20 exact checkpoints; largest numerical gradient error ${maximumError}`);
