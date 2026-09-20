# Medical Specialty Triage System

A web application that classifies English medical descriptions into eight specialties using a fine-tuned biomedical language model.

## Project Status

This repository is being built incrementally to practice software development through small features, tests, and pull requests. It currently contains project documentation and Git ignore rules; application code has not been added yet.

## Planned Scope

- Use existing fine-tuned PubMedBERT weights without retraining.
- Run single-description inference on a local CPU.
- Expose predictions through a FastAPI service.
- Provide a web interface for entering descriptions and viewing results.

## Model and Data

Model weights, medical datasets, and private training notebooks are not included in this repository. Model setup instructions will be added with the inference feature.

## Development Workflow

Each feature will follow an issue, feature branch, implementation, verification, and pull request review before merging into `main`.

## Intended Use

This is an educational prototype, not a clinically validated medical service. Predictions are specialty classifications, not diagnoses or urgency assessments.
