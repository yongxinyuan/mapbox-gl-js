import { normalizePropertyExpression } from "../style-spec/expression/index";
import { StylePropertyExpression } from "../style-spec/expression";
import { StylePropertySpecification } from "../style-spec/style-spec";
import EvaluationParameters from "./evaluation_parameters";
import { CanonicalTileID } from "../source/tile_id";
import {
  TransitionSpecification,
  PropertyValueSpecification,
} from "../style-spec/types";

type TimePoint = number;

export interface CrossFaded<T> {
  to: T;
  from: T;
  other?: T;
}

export interface Property<T, R> {
  specification: StylePropertySpecification;
  possiblyEvaluate: (
    value: PropertyValue<T, R>,
    parameters: EvaluationParameters,
    canonical?: CanonicalTileID,
    availableImages?: Array<string>
  ) => R;
  interpolate: (a: R, b: R, t: number) => R;
}

export class PropertyValue<T, R> {
  property: Property<T, R>;
  value: PropertyValueSpecification<T> | void;
  expression: StylePropertyExpression;

  constructor(
    property: Property<T, R>,
    value: PropertyValueSpecification<T> | void
  ) {
    this.property = property;
    this.value = value;
    this.expression = normalizePropertyExpression(
      value === undefined ? property.specification.default : value,
      property.specification
    );
  }

  isDataDriven(): boolean {
    return (
      this.expression.kind === "source" || this.expression.kind === "composite"
    );
  }
}

// ------- Transitionable -------
class TransitionablePropertyValue<T, R> {}

// ------- Layout -------
type PropertyValues<Props extends Object> = WeakMap<
  Props,
  <T, R>(p: Property<T, R>) => PropertyValue<T, R>
>;

export class Properties<Props> {
  properties: Props;
  defaultPropertyValues: PropertyValues<Props>;
}
