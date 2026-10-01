import { equal, expect, test } from "typroof";

import type { Add, Div, Exp, IsNeg, Mul, Rem, Sub } from "@rivo-ts/math";
import type { Dec as DecNat } from "@rivo-ts/math/Nat/Dec";
import type { Args, Fn, List, Pipe } from "rivo";

type InputNodes = 1;
type HiddenNodes = 4;
type OutputNodes = 1;
type LearningRate = 10;

// Try to fit `y = 0.1x`
type NN = NN.New<InputNodes, HiddenNodes, OutputNodes, LearningRate>;

type test_PredictBefore1 = NN.Predict<NN, [1]>; // Target `[0.1]`
//   ^?
type test_PredictBefore2 = NN.Predict<NN, [2]>; // Target `[0.2]`
//   ^?
type test_PredictBefore3 = NN.Predict<NN, [3]>; // Target `[0.3]`
//   ^?

type TrainedNN = expected_nn20;

// Inspect the predictions after training
type test_PredictAfter1 = NN.Predict<TrainedNN, [1]>; // Target `[0.1]`
//   ^?
type test_PredictAfter2 = NN.Predict<TrainedNN, [2]>; // Target `[0.2]`
//   ^?
type test_PredictAfter3 = NN.Predict<TrainedNN, [3]>; // Target `[0.3]`
//   ^?

/************
 * Training *
 ************/
test("Training", () => {
  expect<NextNN1>().to(equal<expected_nn1>);
  expect<NextNN2>().to(equal<expected_nn2>);
  expect<NextNN3>().to(equal<expected_nn3>);
  expect<NextNN4>().to(equal<expected_nn4>);
  expect<NextNN5>().to(equal<expected_nn5>);
  expect<NextNN6>().to(equal<expected_nn6>);
  expect<NextNN7>().to(equal<expected_nn7>);
  expect<NextNN8>().to(equal<expected_nn8>);
  expect<NextNN9>().to(equal<expected_nn9>);
  expect<NextNN10>().to(equal<expected_nn10>);
  expect<NextNN11>().to(equal<expected_nn11>);
  expect<NextNN12>().to(equal<expected_nn12>);
  expect<NextNN13>().to(equal<expected_nn13>);
  expect<NextNN14>().to(equal<expected_nn14>);
  expect<NextNN15>().to(equal<expected_nn15>);
  expect<NextNN16>().to(equal<expected_nn16>);
  expect<NextNN17>().to(equal<expected_nn17>);
  expect<NextNN18>().to(equal<expected_nn18>);
  expect<NextNN19>().to(equal<expected_nn19>);
  expect<NextNN20>().to(equal<expected_nn20>);
});

test("Zero pre-activations include both sigmoid derivatives", () => {
  expect<NN.Train<NN.New<1, 2, 1, 1, [[0], [0]], [[1, -1]]>, [1], [0]>>().to(
    equal<{
      LearningRate: 1;
      WeightsInputHidden: [[-0.03125], [0.03125]];
      WeightsHiddenOutput: [[0.9375, -1.0625]];
    }>,
  );
});

// Omitting unequal output derivatives reverses this hidden update's sign.
// Exp is approximate at the type level, but this sign matches the numerical gradient.
type test_MultiOutputStep = NN.Train<NN.New<1, 1, 2, 0.01, [[0]], [[6], [1]]>, [1], [0.99, 0.47]>;
type test_HiddenWeight = test_MultiOutputStep["WeightsInputHidden"][0][0];
test("Multi-output hidden update has a concrete negative weight", () => {
  expect<
    [test_HiddenWeight] extends [never] ? false
    : number extends test_HiddenWeight ? false
    : true
  >().to(equal<true>);
  expect<IsNeg<test_HiddenWeight>>().to(equal<true>);
});

type NextNN1 = NN.Train<NN, [1], [0.1]>;
type expected_nn1 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.20141266370581942],
    [-0.3370047335764906],
    [0.06265070149011984],
    [-0.2258420533756396],
  ];
  WeightsHiddenOutput: [
    [-0.5801343531086426, -0.7140657033924445, -0.3165596475361621, -0.6043733407989313],
  ];
};
type NextNN2 = NN.Train<expected_nn1, [2], [0.2]>;
type expected_nn2 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.1471545360987006],
    [-0.2748094993632608],
    [0.09335423957499366],
    [-0.1698950924634186],
  ];
  WeightsHiddenOutput: [
    [-0.6581557919554227, -0.7798115590004283, -0.4200246423654275, -0.6801214778388158],
  ];
};
type NextNN3 = NN.Train<expected_nn2, [1], [0.1]>;
type expected_nn3 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.1068677733953137],
    [-0.2277119303066472],
    [0.11914781375229493],
    [-0.1283385910404999],
  ];
  WeightsHiddenOutput: [
    [-0.7722028408141217, -0.8860915970027704, -0.5488529472929811, -0.7927777330951402],
  ];
};
type NextNN4 = NN.Train<expected_nn3, [2], [0.2]>;
type expected_nn4 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.1023254889351291],
    [-0.2227041558938805],
    [0.12236738977046602],
    [-0.1236986623199634],
  ];
  WeightsHiddenOutput: [
    [-0.7775190999539909, -0.890709405177984, -0.555508180382116, -0.797968003637906],
  ];
};
type NextNN5 = NN.Train<expected_nn4, [1], [0.1]>;
type expected_nn5 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.07445402730071395],
    [-0.1910852238871109],
    [0.1422581197402133],
    [-0.09512866372420087],
  ];
  WeightsHiddenOutput: [
    [-0.845725818585117, -0.954619360365272, -0.6317817942725252, -0.8654090175831249],
  ];
};
type NextNN6 = NN.Train<expected_nn5, [2], [0.2]>;
type expected_nn6 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.08806097798141281],
    [-0.2059789937524149],
    [0.1322409715045514],
    [-0.1090036848286519],
  ];
  WeightsHiddenOutput: [
    [-0.8307497099846866, -0.9414953004303028, -0.6133172533712501, -0.8507649912020219],
  ];
};
type NextNN7 = NN.Train<expected_nn6, [1], [0.1]>;
type expected_nn7 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.06550345070170718],
    [-0.180634549628981],
    [0.14885407813067758],
    [-0.08592648126547199],
  ];
  WeightsHiddenOutput: [
    [-0.8827672658907724, -0.9903229750315437, -0.6713215303070406, -0.9022141699665049],
  ];
};
type NextNN8 = NN.Train<expected_nn7, [2], [0.2]>;
type expected_nn8 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.0863760577107763],
    [-0.2033999504015764],
    [0.1332608647890408],
    [-0.10719315630812941],
  ];
  WeightsHiddenOutput: [
    [-0.8605743792114525, -0.9708201761488454, -0.6440665955515195, -0.880503467879829],
  ];
};
type NextNN9 = NN.Train<expected_nn8, [1], [0.1]>;
type expected_nn9 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.0664169589623021],
    [-0.181073639304121],
    [0.1481601838310268],
    [-0.08679239241596635],
  ];
  WeightsHiddenOutput: [
    [-0.9050406894669013, -1.0125822945387317, -0.6936305711531845, -0.9244872079292292],
  ];
};
type NextNN10 = NN.Train<expected_nn9, [2], [0.2]>;
type expected_nn10 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.09063274107487651],
    [-0.2074134544017176],
    [0.12992230592486514],
    [-0.11145155057414187],
  ];
  WeightsHiddenOutput: [
    [-0.8799482365273975, -0.9905213393715019, -0.662802922318221, -0.899939124400997],
  ];
};
type NextNN11 = NN.Train<expected_nn10, [1], [0.1]>;
type expected_nn11 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.0720762891858293],
    [-0.1867058109228843],
    [0.1438693563986679],
    [-0.09249346164018786],
  ];
  WeightsHiddenOutput: [
    [-0.9202972523232735, -1.028416945689363, -0.7078074146417617, -0.9398493351494601],
  ];
};
type NextNN12 = NN.Train<expected_nn11, [2], [0.2]>;
type expected_nn12 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.09791954211824024],
    [-0.21474661925308886],
    [0.1242975895879394],
    [-0.11879743541344286],
  ];
  WeightsHiddenOutput: [
    [-0.8941007135696372, -1.0053992416137847, -0.6755465785917388, -0.9142252248945775],
  ];
};
type NextNN13 = NN.Train<expected_nn12, [1], [0.1]>;
type expected_nn13 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.08017839849971711],
    [-0.1949779725671385],
    [0.13768247678765347],
    [-0.10067745990846323],
  ];
  WeightsHiddenOutput: [
    [-0.9319347370153978, -1.040924354680664, -0.717795772336044, -0.9516452085574529],
  ];
};
type NextNN14 = NN.Train<expected_nn13, [2], [0.2]>;
type expected_nn14 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.1067959278578738],
    [-0.223790645861121],
    [0.1174352643319458],
    [-0.1277576445458718],
  ];
  WeightsHiddenOutput: [
    [-0.9054889914920762, -1.017713563803708, -0.6851172530491241, -0.9257839181408413],
  ];
};
type NextNN15 = NN.Train<expected_nn14, [1], [0.1]>;
type expected_nn15 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.0895628419673579],
    [-0.20460763038958538],
    [0.1304665130714669],
    [-0.11015992821893617],
  ];
  WeightsHiddenOutput: [
    [-0.9416248453643425, -1.051632245757663, -0.7255282972601329, -0.9615210731587993],
  ];
};
type NextNN16 = NN.Train<expected_nn15, [2], [0.2]>;
type expected_nn16 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.1164951137896761],
    [-0.2336935015700611],
    [0.10990017222479456],
    [-0.13754874739146242],
  ];
  WeightsHiddenOutput: [
    [-0.9153682771011624, -1.028618556390878, -0.692955938471519, -0.9358524547855133],
  ];
};
type NextNN17 = NN.Train<expected_nn16, [1], [0.1]>;
type expected_nn17 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.09960391253676447],
    [-0.2149057148407677],
    [0.12269198908369557],
    [-0.12030259801350708],
  ];
  WeightsHiddenOutput: [
    [-0.9502449004649375, -1.061342333339774, -0.7320199147852838, -0.9703408333922989],
  ];
};
type NextNN18 = NN.Train<expected_nn17, [2], [0.2]>;
type expected_nn18 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.12659059764438998],
    [-0.243983804654143],
    [0.1020088160725796],
    [-0.1477353575142502],
  ];
  WeightsHiddenOutput: [
    [-0.9244099680030738, -1.0387307666341088, -0.6998359694463466, -0.9450924466015396],
  ];
};
type NextNN19 = NN.Train<expected_nn18, [1], [0.1]>;
type expected_nn19 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.10995015717143362],
    [-0.2254870840299829],
    [0.11462435763840265],
    [-0.13074722249350829],
  ];
  WeightsHiddenOutput: [
    [-0.9582718287885, -1.07048963208565, -0.7378247367080619, -0.97857395074177],
  ];
};
type NextNN20 = NN.Train<expected_nn19, [2], [0.2]>;
type expected_nn20 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.1368385637323155],
    [-0.2543939946161432],
    [0.09394316479333577],
    [-0.1580688478121247],
  ];
  WeightsHiddenOutput: [
    [-0.9329820254182816, -1.048387207409663, -0.7061837604547861, -0.9538662849374816],
  ];
};

/******************
 * Implementation *
 ******************/
/**
 * Generate a random number in range [0, 1) using Linear Congruential Generator.
 */
type LCG<Seed extends number> =
  Rem<Add<Mul<1664525, Seed>, 1013904223>, 4294967296> extends infer NextSeed extends number ?
    [Div<NextSeed, 4294967296>, NextSeed]
  : never;

type RandomMatrix<Rows extends number, Cols extends number, CurrSeed extends number = 42> =
  Rows extends 0 ? []
  : _RandomRow<Cols, CurrSeed> extends [infer Row extends number[], infer NextSeed extends number] ?
    [Row, ...RandomMatrix<DecNat<Rows>, Cols, NextSeed>]
  : never;
type _RandomRow<Cols extends number, CurrSeed extends number, Result extends number[] = []> =
  Cols extends 0 ? [Result, CurrSeed]
  : LCG<CurrSeed> extends [infer Value extends number, infer NextSeed extends number] ?
    _RandomRow<DecNat<Cols>, NextSeed, [...Result, Sub<Value, 0.5>]>
  : never;

type Sigmoid<N extends number> = Div<1, Add<1, Exp<Mul<N, -1>>>>;
interface SigmoidFn extends Fn<[number], number> {
  def: ([n]: Args<this>) => Sigmoid<typeof n>;
}
type SigmoidDerivative<N extends number> = Mul<N, Sub<1, N>>;

namespace NN {
  export interface State {
    LearningRate: number;
    WeightsInputHidden: number[][];
    WeightsHiddenOutput: number[][];
  }

  export type New<
    InputNodes extends number,
    HiddenNodes extends number,
    OutputNodes extends number,
    LearningRate extends number,
    WeightsInputHidden extends number[][] = RandomMatrix<HiddenNodes, InputNodes>,
    WeightsHiddenOutput extends number[][] = RandomMatrix<OutputNodes, HiddenNodes>,
  > = {
    LearningRate: LearningRate;
    WeightsInputHidden: WeightsInputHidden;
    WeightsHiddenOutput: WeightsHiddenOutput;
  };

  export type Forward<NN extends State, InputArray extends number[]> =
    Pipe<
      NN["WeightsInputHidden"],
      List.Map<FoldRowWeightFn<InputArray>>,
      List.Map<SigmoidFn>
    > extends infer HiddenOutputs extends number[] ?
      Pipe<
        NN["WeightsHiddenOutput"],
        List.Map<FoldRowWeightFn<HiddenOutputs>>,
        List.Map<SigmoidFn>
      > extends infer FinalOutputs extends number[] ?
        [HiddenOutputs, FinalOutputs]
      : never
    : never;
  interface FoldRowWeightFn<InputArray extends number[]> extends Fn<[number[]], number> {
    def: ([row]: Args<this>) => _FoldRowWeight<typeof row, InputArray>;
  }
  type _FoldRowWeight<
    Row extends number[],
    InputArray extends number[],
    Result extends number = 0,
  > =
    Row extends [infer RowHead extends number, ...infer RowTail extends number[]] ?
      InputArray extends [infer InputHead extends number, ...infer InputTail extends number[]] ?
        _FoldRowWeight<RowTail, InputTail, Add<Result, Mul<RowHead, InputHead>>>
      : never
    : Result;

  export type Train<NN extends State, InputArray extends number[], TargetArray extends number[]> =
    Forward<NN, InputArray> extends (
      [infer HiddenOutputs extends number[], infer FinalOutputs extends number[]]
    ) ?
      CalculateOutputErrors<FinalOutputs, TargetArray> extends infer OutputErrors extends number[] ?
        CalculateOutputDeltas<OutputErrors, FinalOutputs> extends (
          infer OutputDeltas extends number[]
        ) ?
          CalculateHiddenErrors<NN["WeightsHiddenOutput"], OutputDeltas> extends (
            infer HiddenErrors extends number[]
          ) ?
            {
              LearningRate: NN["LearningRate"];
              WeightsInputHidden: UpdateWeights<
                NN["WeightsInputHidden"],
                NN["LearningRate"],
                HiddenErrors,
                HiddenOutputs,
                InputArray
              >;
              WeightsHiddenOutput: UpdateWeights<
                NN["WeightsHiddenOutput"],
                NN["LearningRate"],
                OutputErrors,
                FinalOutputs,
                HiddenOutputs
              >;
            }
          : never
        : never
      : never
    : never;
  type CalculateOutputErrors<
    FinalOutputs extends number[],
    TargetArray extends number[],
    Result extends number[] = [],
  > =
    FinalOutputs extends (
      [infer FinalOutput extends number, ...infer FinalOutputsTail extends number[]]
    ) ?
      TargetArray extends [infer Target extends number, ...infer TargetArrayTail extends number[]] ?
        CalculateOutputErrors<
          FinalOutputsTail,
          TargetArrayTail,
          [...Result, Sub<Target, FinalOutput>]
        >
      : never
    : Result;
  type CalculateOutputDeltas<
    Errors extends number[],
    Outputs extends number[],
    Result extends number[] = [],
  > =
    Errors extends [infer Error extends number, ...infer ErrorsTail extends number[]] ?
      Outputs extends [infer Output extends number, ...infer OutputsTail extends number[]] ?
        CalculateOutputDeltas<
          ErrorsTail,
          OutputsTail,
          [...Result, Mul<Error, SigmoidDerivative<Output>>]
        >
      : never
    : Result;
  type CalculateHiddenErrors<
    WeightsHiddenOutput extends number[][],
    OutputDeltas extends number[],
    I extends number = 0,
    Result extends number[] = [],
  > =
    I extends WeightsHiddenOutput[0]["length"] ? Result
    : CalculateHiddenErrors<
        WeightsHiddenOutput,
        OutputDeltas,
        Add<I, 1>,
        [...Result, _FoldOutputDeltas<WeightsHiddenOutput, I, OutputDeltas>]
      >;
  type _FoldOutputDeltas<
    WeightsHiddenOutput extends number[][],
    Col extends number,
    OutputDeltas extends number[],
    Result extends number = 0,
  > =
    WeightsHiddenOutput extends [infer Row extends number[], ...infer Rows extends number[][]] ?
      OutputDeltas extends (
        [infer OutputDeltaHead extends number, ...infer OutputDeltaTail extends number[]]
      ) ?
        _FoldOutputDeltas<Rows, Col, OutputDeltaTail, Add<Result, Mul<Row[Col], OutputDeltaHead>>>
      : never
    : Result;

  type UpdateWeights<
    Weights extends number[][],
    LearningRate extends number,
    Errors extends number[],
    Outputs extends number[],
    Inputs extends number[],
  > = _UpdateMatrix<LearningRate, Weights, 0, Errors, Outputs, Inputs>;
  type _UpdateWeight<
    Weight extends number,
    LearningRate extends number,
    Error extends number,
    Output extends number,
    Input extends number,
  > = Add<Weight, Mul<Mul<Mul<LearningRate, Error>, SigmoidDerivative<Output>>, Input>>;
  type _UpdateRow<
    LearningRate extends number,
    Row extends number[],
    RowIndex extends number,
    ColIndex extends number,
    Errors extends number[],
    Outputs extends number[],
    Inputs extends number[],
    Result extends number[] = [],
  > =
    ColIndex extends Row["length"] ? Result
    : _UpdateWeight<
      Row[ColIndex],
      LearningRate,
      Errors[RowIndex],
      Outputs[RowIndex],
      Inputs[ColIndex]
    > extends infer UpdatedWeight extends number ?
      _UpdateRow<
        LearningRate,
        Row,
        RowIndex,
        Add<ColIndex, 1>,
        Errors,
        Outputs,
        Inputs,
        [...Result, UpdatedWeight]
      >
    : never;
  type _UpdateMatrix<
    LearningRate extends number,
    Matrix extends number[][],
    RowIndex extends number,
    Errors extends number[],
    Outputs extends number[],
    Inputs extends number[],
    Result extends number[][] = [],
  > =
    RowIndex extends Matrix["length"] ? Result
    : _UpdateRow<LearningRate, Matrix[RowIndex], RowIndex, 0, Errors, Outputs, Inputs> extends (
      infer UpdatedRow extends number[]
    ) ?
      _UpdateMatrix<
        LearningRate,
        Matrix,
        Add<RowIndex, 1>,
        Errors,
        Outputs,
        Inputs,
        [...Result, UpdatedRow]
      >
    : never;

  export type Predict<NN extends State, InputArray extends number[]> = Forward<NN, InputArray>[1];
}
