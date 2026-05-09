# Code Rules

This file defines the coding rules for the project. These rules must be followed across the backend, frontend, and any shared scripts.

## 1) Use enums when possible
- Use enums to represent fixed sets of values (status, categories, label values, UI modes).
- Prefer enums over raw strings or magic numbers in business logic and API payloads.
- Ensure enums are centralized and reused across modules.
- When exchanging data with the frontend, map enums to stable string values.

## 2) Layered architecture is mandatory
Follow a clear separation of concerns. Each layer must have a single responsibility.

- Presentation/UI layer: screens, components, and UI-only state.
- Controller/API layer: request parsing, validation, response shaping.
- Service layer: business logic and orchestration.
- Data access layer: database or file I/O, queries, persistence.
- Model/ML layer: model loading, inference, vectorization, and similarity calculation.

Rules:
- Do not mix logic across layers. For example, the UI must not perform model inference.
- Controller/API layer must not contain business logic beyond validation and routing.
- Service layer is the only place where multiple data sources or models are combined.
- Data access must be isolated to dedicated modules or classes.

## 3) Naming and structure
- Keep module and file names descriptive and consistent.
- Use explicit names for services and repositories (for example, `ProductService`, `ReviewRepository`).
- Avoid deep nesting and circular dependencies.

## 4) Error handling
- All API endpoints must return structured error responses.
- Validate inputs at the API layer and raise domain-specific errors in services.
- Log errors with enough context to reproduce issues.

## 5) Configuration
- Store configuration in a single place and avoid hard-coded paths or secrets.
- Paths for datasets and model artifacts must be configurable.

## 6) Documentation
- Each public function or endpoint must have a short docstring or comment explaining purpose and inputs/outputs.
- Keep README and context references up to date as implementation evolves.
