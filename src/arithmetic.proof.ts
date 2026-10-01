import { equal, expect, test } from "typroof";

import type { Add, Mul, Sub } from "@rivo-ts/math";
import type { _Div10 } from "@rivo-ts/math/internals/_Div10";
import type { _ReverseString } from "@rivo-ts/math/internals/_ReverseString";

test("Patched decimal arithmetic", () => {
  expect<_ReverseString<"">>().to(equal<"">);
  expect<_ReverseString<"1">>().to(equal<"1">);
  expect<_ReverseString<"-0.123456789">>().to(equal<"987654321.0-">);
  expect<_ReverseString<"1.25e-8">>().to(equal<"8-e52.1">);
  expect<
    _ReverseString<"0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789">
  >().to(
    equal<"9876543210987654321098765432109876543210987654321098765432109876543210987654321098765432109876543210987654321098765432109876543210">,
  );
  expect<Add<0.5, -0.5>>().to(equal<0>);
  expect<Add<-0.5, 0.5>>().to(equal<0>);
  expect<Add<1.23, -1.23>>().to(equal<0>);
  expect<Add<-1.23, 1.23>>().to(equal<0>);
  expect<Add<0.125, -0.125>>().to(equal<0>);
  expect<Add<5, -5>>().to(equal<0>);
  expect<Add<-5, 5>>().to(equal<0>);
  expect<Sub<0.5, 0.5>>().to(equal<0>);
  expect<Add<0, 0>>().to(equal<0>);
  expect<Mul<0.5, 0>>().to(equal<0>);
  expect<_Div10<"0", "_">>().to(equal<"0">);
  expect<_Div10<"0", "_____">>().to(equal<"0">);
  expect<_Div10<"-0", "_">>().to(equal<"0">);
  expect<_Div10<"0", "">>().to(equal<"0">);
  expect<_Div10<"125", "__">>().to(equal<"1.25">);
  expect<_Div10<"-125", "__">>().to(equal<"-1.25">);
});
