export interface PredictionInput {
  features: number[];
}

export interface PredictionOutput {
  prediction: number;
  confidence?: number;
}
