import { equal, expect, test } from "typroof";

import type { Add, Div, Exp, Mul, Rem, Sub } from "@rivo-ts/math";
import type { Dec as DecNat } from "@rivo-ts/math/Nat/Dec";
import type { Args, Fn, List, Pipe } from "rivo";

type InputNodes = 1;
type HiddenNodes = 4;
type OutputNodes = 1;
type LearningRate = 10;

// Try to fit `y = 0.1x`
type NN = NN.New<InputNodes, HiddenNodes, OutputNodes, LearningRate>;

type test_PredictBefore1 = NN.Predict<NN, [1]>; // Expected `[0.1]`
//   ^?
type test_PredictBefore2 = NN.Predict<NN, [2]>; // Expected `[0.2]`
//   ^?
type test_PredictBefore3 = NN.Predict<NN, [3]>; // Expected `[0.3]`
//   ^?

type TrainedNN = expected_nn20;

// The result gets better after training
type test_PredictAfter1 = NN.Predict<TrainedNN, [1]>; // Expected `[0.1]`
//   ^?
type test_PredictAfter2 = NN.Predict<TrainedNN, [2]>; // Expected `[0.2]`
//   ^?
type test_PredictAfter3 = NN.Predict<TrainedNN, [3]>; // Expected `[0.3]`
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

type NextNN1 = NN.Train<NN, [1], [0.1]>;
type expected_nn1 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.05689195999725226],
    [-0.1030127082451784],
    [0.01692598587106627],
    [-0.0645649850885663],
  ];
  WeightsHiddenOutput: [
    [-0.5801343531086426, -0.7140657033924445, -0.3165596475361621, -0.6043733407989313],
  ];
};
type NextNN2 = NN.Train<expected_nn1, [2], [0.2]>;
type expected_nn2 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [0.121235528169096],
    [0.1146304488353882],
    [0.1144111253884506],
    [0.12083232343522042],
  ];
  WeightsHiddenOutput: [
    [-0.6362565954333449, -0.7674615230024028, -0.37707059456269, -0.6600406939108527],
  ];
};
type NextNN3 = NN.Train<expected_nn2, [1], [0.1]>;
type expected_nn3 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [0.3042153200427214],
    [0.3354290888636164],
    [0.2228956766249617],
    [0.3106567638546999],
  ];
  WeightsHiddenOutput: [
    [-0.7397478428991743, -0.8706316185823187, -0.4802300240769713, -0.7635123403975798],
  ];
};
type NextNN4 = NN.Train<expected_nn3, [2], [0.2]>;
type expected_nn4 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [0.089850838955252],
    [0.0879621598571738],
    [0.0777829477046886],
    [0.09025385747219422],
  ];
  WeightsHiddenOutput: [
    [-0.6912847692746977, -0.8211124004347462, -0.434606610181497, -0.7148296583676944],
  ];
};
type NextNN5 = NN.Train<expected_nn4, [1], [0.1]>;
type expected_nn5 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [0.2615659722220671],
    [0.2919435898788503],
    [0.1857938076013978],
    [0.2678143211493893],
  ];
  WeightsHiddenOutput: [
    [-0.7743719309293082, -0.9041246192091773, -0.517214813311722, -0.7979328109969541],
  ];
};
type NextNN6 = NN.Train<expected_nn5, [2], [0.2]>;
type expected_nn6 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [0.01855745063648948],
    [0.0128311090137407],
    [0.01798412052725124],
    [0.01822021477685276],
  ];
  WeightsHiddenOutput: [
    [-0.7257984768349929, -0.8544618050442007, -0.4714292429055537, -0.7491338406846675],
  ];
};
type NextNN7 = NN.Train<expected_nn6, [1], [0.1]>;
type expected_nn7 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [0.1925727927508995],
    [0.2177036043823133],
    [0.1310132242898422],
    [0.19783093308723426],
  ];
  WeightsHiddenOutput: [
    [-0.8020437605441879, -0.9304907928582146, -0.5476528365569244, -0.8253663873519441],
  ];
};
type NextNN8 = NN.Train<expected_nn7, [2], [0.2]>;
type expected_nn8 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.05301975611375504],
    [-0.0643381400081061],
    [-0.04002618695929318],
    [-0.05439140101266984],
  ];
  WeightsHiddenOutput: [
    [-0.7574843955282112, -0.8850291409310862, -0.505338178274527, -0.780617484978352],
  ];
};
type NextNN9 = NN.Train<expected_nn8, [1], [0.1]>;
type expected_nn9 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [0.12484328190797196],
    [0.1434043894518332],
    [0.0786668834114672],
    [0.12889671129317526],
  ];
  WeightsHiddenOutput: [
    [-0.8290162854570379, -0.956145569264866, -0.577347180042505, -0.852099018672757],
  ];
};
type NextNN10 = NN.Train<expected_nn9, [2], [0.2]>;
type expected_nn10 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.1145212411833026],
    [-0.1313036400824958],
    [-0.089600623675878],
    [-0.11688099787975556],
  ];
  WeightsHiddenOutput: [
    [-0.7890034264056461, -0.915483843541487, -0.538960674315576, -0.8119441872187045],
  ];
};
type NextNN11 = NN.Train<expected_nn10, [1], [0.1]>;
type expected_nn11 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [0.0656577410480681],
    [0.077543518461725],
    [0.033634392371339],
    [0.06851151743089931],
  ];
  WeightsHiddenOutput: [
    [-0.8559299660565689, -0.9818169709485335, -0.6067694176086965, -0.8787872512683554],
  ];
};
type NextNN12 = NN.Train<expected_nn11, [2], [0.2]>;
type expected_nn12 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.16530833124311048],
    [-0.1869424160195306],
    [-0.1306186056402688],
    [-0.1685317748983867],
  ];
  WeightsHiddenOutput: [
    [-0.819965496654864, -0.9454533979156068, -0.5718830451154425, -0.8427268957184624],
  ];
};
type NextNN13 = NN.Train<expected_nn12, [1], [0.1]>;
type expected_nn13 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [0.01540933416825332],
    [0.02103678203172024],
    [-0.00425419479855717],
    [0.017152581703187],
  ];
  WeightsHiddenOutput: [
    [-0.882322321870103, -1.007080761774596, -0.635412202266653, -0.9049749456980313],
  ];
};
type NextNN14 = NN.Train<expected_nn13, [2], [0.2]>;
type expected_nn14 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.20740846541066588],
    [-0.23323481573621246],
    [-0.1647535985087625],
    [-0.2113728408672512],
  ];
  WeightsHiddenOutput: [
    [-0.8497134259253576, -0.974291207438022, -0.6034347402323982, -0.8723100821274483],
  ];
};
type NextNN15 = NN.Train<expected_nn14, [1], [0.1]>;
type expected_nn15 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.02745885396322098],
    [-0.02748634226016472],
    [-0.0364534164855989],
    [-0.02671416090850907],
  ];
  WeightsHiddenOutput: [
    [-0.9077447353178921, -1.0314968636296957, -0.6628344067688272, -0.930214502669339],
  ];
};
type NextNN16 = NN.Train<expected_nn15, [2], [0.2]>;
type expected_nn16 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.24281323652666226],
    [-0.2721994495146932],
    [-0.1936146548408096],
    [-0.247408197026739],
  ];
  WeightsHiddenOutput: [
    [-0.877899471472189, -1.001652442678084, -0.6332648870988544, -0.9003464032425499],
  ];
};
type NextNN17 = NN.Train<expected_nn16, [1], [0.1]>;
type expected_nn17 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.06449850410854206],
    [-0.06951269275198317],
    [-0.0642993977059275],
    [-0.0646365926170777],
  ];
  WeightsHiddenOutput: [
    [-0.9319620005702387, -1.054826305479481, -0.6888221214717782, -0.9542697584710337],
  ];
};
type NextNN18 = NN.Train<expected_nn17, [2], [0.2]>;
type expected_nn18 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.2730089604295769],
    [-0.3053536336653403],
    [-0.2184154377743168],
    [-0.2781342239874653],
  ];
  WeightsHiddenOutput: [
    [-0.9044220723323666, -1.02743331149586, -0.6612763566627227, -0.9267338780478881],
  ];
};
type NextNN19 = NN.Train<expected_nn18, [1], [0.1]>;
type expected_nn19 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.09689676534719033],
    [-0.1062152883327709],
    [-0.08878726951325486],
    [-0.0978039870782199],
  ];
  WeightsHiddenOutput: [
    [-0.9548949226767359, -1.07698125927322, -0.7133192757466533, -0.9770598908460446],
  ];
};
type NextNN20 = NN.Train<expected_nn19, [2], [0.2]>;
type expected_nn20 = {
  LearningRate: 10;
  WeightsInputHidden: [
    [-0.299082648025036],
    [-0.3338215583844103],
    [-0.2400497748481249],
    [-0.3046465731474277],
  ];
  WeightsHiddenOutput: [
    [-0.9293062758093199, -1.051653850305836, -0.687502900361073, -0.9514966986717136],
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
        CalculateHiddenErrors<NN["WeightsHiddenOutput"], OutputErrors> extends (
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
  type CalculateHiddenErrors<
    WeightsHiddenOutput extends number[][],
    OutputErrors extends number[],
    I extends number = 0,
    Result extends number[] = [],
  > =
    I extends WeightsHiddenOutput[0]["length"] ? Result
    : CalculateHiddenErrors<
        WeightsHiddenOutput,
        OutputErrors,
        Add<I, 1>,
        [...Result, _FoldOutputErrors<WeightsHiddenOutput, I, OutputErrors>]
      >;
  type _FoldOutputErrors<
    WeightsHiddenOutput extends number[][],
    Col extends number,
    OutputErrors extends number[],
    Result extends number = 0,
  > =
    WeightsHiddenOutput extends [infer Row extends number[], ...infer Rows extends number[][]] ?
      OutputErrors extends (
        [infer OutputErrorHead extends number, ...infer OutputErrorTail extends number[]]
      ) ?
        _FoldOutputErrors<Rows, Col, OutputErrorTail, Add<Result, Mul<Row[Col], OutputErrorHead>>>
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
