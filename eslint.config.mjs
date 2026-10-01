import js from "@eslint/js";
import tsPlugin from "@typescript-eslint/eslint-plugin";
import * as tsParser from "@typescript-eslint/parser";
import { createTypeScriptImportResolver } from "eslint-import-resolver-typescript";
import { flatConfigs as importConfigs } from "eslint-plugin-import-x";
import prettier from "eslint-plugin-prettier/recommended";
import { configs as sonarConfigs } from "eslint-plugin-sonarjs";
import sortDestructureKeys from "eslint-plugin-sort-destructure-keys";

export default [
  { ignores: ["node_modules/**"] },
  js.configs.recommended,
  importConfigs.recommended,
  importConfigs.typescript,
  sonarConfigs.recommended,
  prettier,
  {
    files: ["**/*.{js,cjs,mjs,ts}"],
    languageOptions: {
      globals: { console: "readonly", module: "readonly", process: "readonly" },
    },
    plugins: { "sort-destructure-keys": sortDestructureKeys },
    settings: {
      "import-x/resolver-next": [createTypeScriptImportResolver({ alwaysTryTypes: true })],
    },
    rules: {
      "import-x/export": "off",
      "import-x/consistent-type-specifier-style": ["error", "prefer-top-level"],
      "import-x/namespace": "off",
      "import-x/no-unresolved": "off",
      "import-x/order": [
        "error",
        {
          alphabetize: { order: "asc" },
          groups: [
            "builtin",
            "external",
            "internal",
            "parent",
            "sibling",
            "index",
            "object",
            "type",
          ],
          "newlines-between": "always",
        },
      ],
      "no-undef": "off",
      "sonarjs/cognitive-complexity": "off",
      "sort-destructure-keys/sort-destructure-keys": "error",
      "sort-imports": [
        "error",
        {
          ignoreCase: false,
          ignoreDeclarationSort: true,
          ignoreMemberSort: false,
          memberSyntaxSortOrder: ["none", "all", "multiple", "single"],
        },
      ],
    },
  },
  {
    files: ["**/*.ts"],
    languageOptions: { parser: tsParser },
    plugins: { "@typescript-eslint": tsPlugin },
    rules: {
      ...tsPlugin.configs.recommended.rules,
      "no-redeclare": "off",
      "@typescript-eslint/consistent-type-imports": [
        "error",
        { prefer: "type-imports", disallowTypeAnnotations: false },
      ],
      "@typescript-eslint/no-namespace": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        { vars: "all", varsIgnorePattern: "test_.*", argsIgnorePattern: "_" },
      ],
    },
  },
  {
    files: ["src/main.ts"],
    rules: { "sonarjs/redundant-type-aliases": "off" },
  },
];
